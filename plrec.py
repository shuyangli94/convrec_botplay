import json
from datetime import datetime

import torch
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import LinearRegression

from convrec.utils import _load, _save

NEG_CRITIQUE_PENALTY = 100


class PLRecWithAspects():
    def __init__(self, **kwargs):

        # Override arguments
        self.train_kwargs = {'n_iter': 7, 'lamb': 80, 'dim': 50, 'seed': 1}
        for k in list(kwargs.keys()):
            if k not in self.train_kwargs:
                continue
            self.train_kwargs[k] = kwargs.pop(k)

        # Save attributes
        self.n_iter = self.train_kwargs['n_iter']  # Number of power iterations
        self.lamb = self.train_kwargs['lamb']  # Penalty parameter
        self.dim = self.train_kwargs['dim']  # Latent (embedding) dimension
        self.seed = self.train_kwargs['seed']  # Random seed

        if kwargs:
            print('!! Unused kwargs:\n{}'.format(kwargs))

        # Learned embeddings
        self._user_emb = None
        self._item_emb = None
        self._item_bias = None
        self._aspect_encoder = None
        self.n_items = None
        self.n_users = None

        print('Created {} with arguments:\n{}'.format(
            self.__class__.__name__,
            json.dumps(self.train_kwargs, indent=2, default=str)))

    def save_weights(self, path: str):
        if any(p is None for p in
               [self._user_emb, self._item_emb, self._aspect_encoder]):
            raise NotImplementedError(
                'PLRec needs to be trained or weights loaded')
        _save((self.train_kwargs, self._user_emb, self._item_emb,
               self._aspect_encoder), path, 'PLRec weights')

    @classmethod
    def from_pretrained(cls, path: str):
        # Load weights
        train_kwargs, user_emb, item_emb, aspect_encoder = _load(
            path, 'PLRec saved weights')
        model = cls(**train_kwargs)
        model._user_emb = user_emb
        model._item_emb = item_emb
        model._aspect_encoder = aspect_encoder

        # Item/user sizes
        model.n_items = item_emb.shape[0]
        model.n_users = user_emb.shape[0]
        return model

    @property
    def user_emb(self):
        if self._user_emb is None:
            raise NotImplementedError(
                'PLRec needs to be trained or weights loaded!')
        return self._user_emb

    @property
    def item_emb(self):
        if self._item_emb is None:
            raise NotImplementedError(
                'PLRec needs to be trained or weights loaded!')
        return self._item_emb

    @property
    def aspect_encoder(self):
        if self._aspect_encoder is None:
            raise NotImplementedError(
                'PLRec needs to be trained or weights loaded!')
        return self._aspect_encoder

    def get_scores(self,
                   u,
                   latent_u=None,
                   invalid=None,
                   neg_critiques=None,
                   return_latent: bool = False,
                   *args,
                   **kwargs):
        # B x K
        if latent_u is None:
            if isinstance(u, torch.Tensor):
                u = u.numpy()
            latent_u = self.user_emb[u]

        # Item latent * User latent -> B x I
        item_scores = latent_u.dot(self.item_emb.T)

        # Add item bias of shape I,
        if self._item_bias is not None:
            item_scores = item_scores + self._item_bias

        # Downrank filtered items
        item_scores = self.downweight_neg_items(item_scores, neg_critiques)

        # Remove invalid items (list of tensor indices)
        item_scores, n_candidates = self.filter_invalid(item_scores, invalid)

        if return_latent:
            return item_scores, n_candidates, latent_u

        return item_scores, n_candidates

    @classmethod
    def downweight_neg_items(cls, scores, neg_critiques):
        if neg_critiques is not None:
            for ix, u_neg_crit in enumerate(neg_critiques):
                scores[
                    ix,
                    u_neg_crit] = scores[ix, u_neg_crit] - NEG_CRITIQUE_PENALTY
        return scores

    @classmethod
    def filter_invalid(cls, scores, invalid):
        # Remove invalid items (list of tensor indices)
        batch_size = scores.shape[0]
        n_items = scores.shape[-1]
        n_candidates = [n_items] * batch_size
        if invalid is not None:
            for ix, u_inv in enumerate(invalid):
                scores[ix, u_inv] = -np.inf
                n_candidates[ix] -= len(u_inv)

        return scores, n_candidates

    def get_ranks(self,
                  u,
                  item_scores=None,
                  index_by_item: bool = True,
                  *args,
                  **kwargs):
        # B x I
        if item_scores is None:
            item_scores, *_ = self.get_scores(u, *args, **kwargs)

        # item_ranks[b, i] = i-ranked item ["item in rank"]
        # Multiply by -1 to do descending
        item_ranks = np.argsort(-1 * item_scores, axis=-1)
        if index_by_item:
            # item_ranks[b, i] = rank of item i ["rank of item"]
            item_ranks = np.argsort(item_ranks, axis=-1)

        return item_ranks

    def train_aspect_encoder(self, train_u_kp_freq: np.array):
        start = datetime.now()

        # Aspect co-embedding
        self._aspect_encoder = LinearRegression().fit(train_u_kp_freq,
                                                      self._user_emb)
        print('{} - Trained linear regression with weights {} and bias {}'.
              format(datetime.now() - start, self._aspect_encoder.coef_.shape,
                     self._aspect_encoder.intercept_.shape))
        return self

    def train_plrec(self, train_mat: np.array):
        # SVD
        # P : U x E
        # Sigma: E
        # Qt : E x I
        start = datetime.now()
        P, sigma, Qt = randomized_svd(
            train_mat,
            n_components=self.dim,
            n_iter=self.n_iter,
            random_state=self.seed)
        print('{} - Performed randomized SVD M = P{} Sigma{} Qt{}'.format(
            datetime.now() - start, P.shape, sigma.shape, Qt.shape))

        # Recover RQ (User embedding): U x E
        train_mat_sparse = sparse.csc_matrix(train_mat)
        RQ = train_mat_sparse.dot(sparse.csc_matrix(Qt.T * np.sqrt(sigma)))
        print('{} - Recovered RQ{} from randomized SVD'.format(
            datetime.now() - start, RQ.shape))

        # Linear optimization
        pre_inv = RQ.T.dot(RQ) + self.lamb * sparse.identity(
            self.dim, dtype=np.float32)
        print('{} - Computed pre-inverse{} from RQ & lambda'.format(
            datetime.now() - start, pre_inv.shape))
        inverse = sparse.linalg.inv(pre_inv.tocsc())
        print('{} - Computed inverse{} from pre-inverse'.format(
            datetime.now() - start, inverse.shape))
        # Y (transposed item weights): E x I
        Y = inverse.dot(RQ.T).dot(train_mat_sparse)
        print('{} - Found Y{} for PLRec'.format(datetime.now() - start,
                                                Y.shape))

        # Set weights
        # User emb: U x E
        self._user_emb = np.array(RQ.todense())
        # Item emb: I x E
        self._item_emb = np.array(Y.T.todense())

        # Save user and item numbers
        self.n_users, self.n_items = train_mat.shape

        return self
