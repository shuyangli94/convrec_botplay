import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.extmath import randomized_svd

from convrec.utils import _load, _save, count_parameters

NEG_CRITIQUE_PENALTY = 100
CRITIQUE_WEIGHTING = 100


class PLRecWithAspectsKP:
    """
    PLRec - https://ssanner.github.io/papers/anu/ijcai16_proj_lrec.pdf

    Trained via SVD
    """

    def __init__(self, n_kp: int, **kwargs):

        # Override arguments
        self.train_kwargs = {
            "n_iter": 7,
            "lamb": 80,
            "dim": 50,
            "seed": 1,
            "n_kp": n_kp,
            "use_user_emb": True,
        }
        for k in list(kwargs.keys()):
            if k not in self.train_kwargs:
                continue
            self.train_kwargs[k] = kwargs.pop(k)

        # Save attributes
        self.n_iter = self.train_kwargs["n_iter"]  # Number of power iterations
        self.lamb = self.train_kwargs["lamb"]  # Penalty parameter
        self.dim = self.train_kwargs["dim"]  # Latent (embedding) dimension
        self.seed = self.train_kwargs["seed"]  # Random seed
        self.use_user_emb = self.train_kwargs[
            "use_user_emb"
        ]  # Use user embedding alongside Aspect Encoder
        self.n_kp = n_kp

        if kwargs:
            print("!! Unused kwargs:\n{}".format(kwargs))

        # Learned embeddings
        self._user_emb = None
        self._item_emb = None
        self._item_bias = None
        self._aspect_encoder = None
        self.n_items = None
        self.n_users = None

        # Placeholders for gold
        self._kp_proj = None

        print(
            "Created {} with arguments:\n{}".format(
                self.__class__.__name__,
                json.dumps(self.train_kwargs, indent=2, default=str),
            )
        )

    def save_weights(self, path: str):
        if any(
            p is None
            for p in [
                self._user_emb,
                self._item_emb,
                self._aspect_encoder,
                self._kp_proj,
            ]
        ):
            raise NotImplementedError("PLRec needs to be trained or weights loaded")
        _save(
            (
                self.train_kwargs,
                self._user_emb,
                self._item_emb,
                self._aspect_encoder,
                self._kp_proj,
            ),
            path,
            "PLRec weights",
        )

    @classmethod
    def from_pretrained(cls, path: str):
        start = datetime.now()
        # Load weights
        train_kwargs, user_emb, item_emb, aspect_encoder, kp_proj = _load(
            path, "PLRec saved weights"
        )
        model = cls(**train_kwargs)
        model._user_emb = user_emb
        model._item_emb = item_emb
        model._aspect_encoder = aspect_encoder
        model._kp_proj = kp_proj

        # Item/user sizes
        model.n_items = item_emb.shape[0]
        model.n_users = user_emb.shape[0]

        print("{} - loaded pretrained weights".format(datetime.now() - start))
        return model

    @property
    def user_emb(self):
        if self._user_emb is None:
            raise NotImplementedError("PLRec needs to be trained or weights loaded!")
        return self._user_emb

    @property
    def item_emb(self):
        if self._item_emb is None:
            raise NotImplementedError("PLRec needs to be trained or weights loaded!")
        return self._item_emb

    @property
    def aspect_encoder(self):
        if self._aspect_encoder is None:
            raise NotImplementedError("PLRec needs to be trained or weights loaded!")
        return self._aspect_encoder

    @property
    def kp_proj(self):
        if self._kp_proj is None:
            raise NotImplementedError("PLRec needs to be trained or weights loaded!")
        return self._kp_proj

    def encode_user(
        self, u: np.array, user_kps: np.array, user_embedding: np.array = None
    ):
        # User latent factors - B x |K| -> B x E
        latent_u = self.aspect_encoder.predict(user_kps)
        if self.use_user_emb:
            if user_embedding is None:
                user_embedding = self.user_emb[u]
            latent_u = (latent_u + user_embedding) * 0.5
        return latent_u, user_embedding

    def get_scores(
        self,
        u,
        user_kps,
        latent_u=None,
        user_embedding=None,
        invalid=None,
        neg_critiques=None,
        return_latent: bool = False,
        return_embedding: bool = False,
        *args,
        **kwargs
    ):
        # B x K
        if latent_u is None:
            latent_u, user_embedding = self.encode_user(
                u, user_kps, user_embedding=user_embedding
            )

        # Item latent * User latent -> B x I
        item_scores = latent_u.dot(self.item_emb.T)

        # Downrank filtered items
        item_scores = self.downweight_neg_items(item_scores, neg_critiques)

        # Remove invalid items (list of tensor indices)
        item_scores, n_candidates = self.filter_invalid(item_scores, invalid)

        returns = [item_scores, n_candidates]

        if return_latent:
            returns.append(latent_u)

        if return_embedding:
            returns.append(user_embedding)

        return returns

    @classmethod
    def downweight_neg_items(cls, scores, neg_critiques):
        if neg_critiques is not None:
            for ix, u_neg_crit in enumerate(neg_critiques):
                scores[ix, u_neg_crit] = scores[ix, u_neg_crit] - NEG_CRITIQUE_PENALTY
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

    def get_ranks(
        self, u, item_scores=None, index_by_item: bool = True, *args, **kwargs
    ):
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
        # train_u_kp_freq : U x K
        # user_emb : U x E
        self._aspect_encoder = LinearRegression().fit(train_u_kp_freq, self._user_emb)
        print(
            "{} - Trained linear regression with weights {} and bias {}".format(
                datetime.now() - start,
                self._aspect_encoder.coef_.shape,
                self._aspect_encoder.intercept_.shape,
            )
        )
        return self

    def prepare_kp_train(self, train_ui: dict, item_kp_freq_mat: np.array):
        # train_ui: Nu-size dict, of u : i_s (positive training examples)
        # item_kp_freq: Ni x k boolean matrix of whether KPs appear in an item's reviews
        start = datetime.now()
        train_inputs = []
        train_targets = []
        for u, i_s in train_ui.items():
            for i in i_s:
                train_inputs.append(self.user_emb[u] + self.item_emb[i])
                train_targets.append(item_kp_freq_mat[i])

        train_inputs = np.stack(train_inputs)  # |R+| x E
        train_targets = np.stack(train_targets)  # |R+| x K

        # Trivial 1 prediction
        for kp_ix, is_trivial in enumerate(
            train_targets.sum(axis=0) == len(train_targets)
        ):
            if is_trivial:
                train_targets[0][kp_ix] = 0

        # Trivial 0 prediction
        for kp_ix, is_trivial in enumerate(train_targets.sum(axis=0) == 0):
            if is_trivial:
                train_targets[0][kp_ix] = 1

        print(
            "{} - Created {} training inputs and {} training targets for KP projection".format(
                datetime.now() - start, train_inputs.shape, train_targets.shape
            )
        )
        return train_inputs, train_targets

    def train_kp_proj(
        self, train_inputs: np.array, train_targets: np.array, max_iter: int = 5000
    ):
        # Train inputs: |R+| x E (User/item embeddings)
        # Train targets: |R+| x K (boolean)
        # Train a multi-label Logistic Regression (equivalent to BCE loss)
        start = datetime.now()

        assert train_targets.shape[1] == self.n_kp

        self._kp_proj = MultiOutputClassifier(
            estimator=LogisticRegression(max_iter=max_iter), n_jobs=-1
        ).fit(train_inputs, train_targets.astype(int))

        print(
            "{} - Trained multi-label logistic regression to predict keywords".format(
                datetime.now() - start
            )
        )
        return self

    def train_plrec(self, train_mat: np.array):
        # SVD
        # train_mat : U x I
        # P : U x E
        # Sigma: E
        # Qt : E x I
        start = datetime.now()
        P, sigma, Qt = randomized_svd(
            train_mat, n_components=self.dim, n_iter=self.n_iter, random_state=self.seed
        )
        print(
            "{} - Performed randomized SVD M = P{} Sigma{} Qt{}".format(
                datetime.now() - start, P.shape, sigma.shape, Qt.shape
            )
        )

        # Recover RQ (User embedding): U x E
        train_mat_sparse = sparse.csc_matrix(train_mat)
        RQ = train_mat_sparse.dot(sparse.csc_matrix(Qt.T * np.sqrt(sigma)))
        print(
            "{} - Recovered RQ{} from randomized SVD".format(
                datetime.now() - start, RQ.shape
            )
        )

        # Linear optimization
        pre_inv = RQ.T.dot(RQ) + self.lamb * sparse.identity(self.dim, dtype=np.float32)
        print(
            "{} - Computed pre-inverse{} from RQ & lambda".format(
                datetime.now() - start, pre_inv.shape
            )
        )
        inverse = sparse.linalg.inv(pre_inv.tocsc())
        print(
            "{} - Computed inverse{} from pre-inverse".format(
                datetime.now() - start, inverse.shape
            )
        )
        # Y (transposed item weights): E x I
        Y = inverse.dot(RQ.T).dot(train_mat_sparse)
        print("{} - Found Y{} for PLRec".format(datetime.now() - start, Y.shape))

        # Set weights
        # User emb: U x E
        self._user_emb = np.array(RQ.todense())
        # Item emb: I x E
        self._item_emb = np.array(Y.T.todense())

        # Save user and item numbers
        self.n_users, self.n_items = train_mat.shape

        return self


class PLRecWithAspectsKPTorch(nn.Module):
    def __init__(
        self,
        k: int,
        n_items: int,
        n_users: int,
        n_kp: int,
        user_emb: bool = False,
        **kwargs
    ):
        super().__init__()

        # Params
        self.k = k
        self.n_items = n_items
        self.n_users = n_users
        self.n_kp = n_kp
        self.use_user_emb = user_emb

        print(
            "Unused initialization arguments for {}:\n{}".format(
                self.__class__.__name__, json.dumps(kwargs, indent=2, default=str)
            )
        )

        self.train_kwargs = {
            "k": self.k,
            "n_items": self.n_items,
            "n_users": self.n_users,
            "n_kp": self.n_kp,
            "enc_layers": 1,
            "proj_layers": 1,
            "user_emb": self.use_user_emb,
        }

        # Latent representations
        self.gamma_i = nn.Embedding(self.n_items, k)
        self.gamma_u = None
        if self.use_user_emb:
            self.gamma_u = nn.Embedding(self.n_users, k)
            print("> Using {} user embedding".format(self.gamma_u.weight.shape))

        # Aspect encoder
        self.kp_encoder = nn.Linear(self.n_kp, self.k)
        print("> Encoding aspects with Linear projection")

        # Aspect projection - proj_layers == -1 means don't project
        self.kp_proj = nn.Linear(self.k, self.n_kp)
        print("> Predicting aspects with Linear head")

        print(
            "Created {} for {:,} users, {:,} items, {:,} aspects, k={} with {:,} parameters".format(
                self.__class__.__name__,
                self.n_users,
                self.n_items,
                self.n_kp,
                self.k,
                count_parameters(self),
            )
        )

    @classmethod
    def from_pretrained_numpy(cls, path: str):
        # Load numpy model from path
        np_model = PLRecWithAspectsKP.from_pretrained(path)

        # Create model with same specifications in torch
        model = cls(
            k=np_model.dim,
            n_items=np_model.n_items,
            n_users=np_model.n_users,
            n_kp=np_model.n_kp,
            user_emb=np_model.use_user_emb,
        )

        ############################
        # Item embeddings
        _ = model.gamma_i.weight.data.copy_(torch.FloatTensor(np_model._item_emb))

        ############################
        # User embeddings
        if model.use_user_emb:
            _ = model.gamma_u.weight.data.copy_(torch.FloatTensor(np_model._user_emb))

        ############################
        # Aspect encoder

        # Set aspect encoder bias term
        np_asp_bias = torch.FloatTensor(np_model.aspect_encoder.intercept_)
        _ = model.kp_encoder.bias.data.copy_(np_asp_bias.data)

        # Set aspect encoder weight term
        np_asp_weight = torch.FloatTensor(np_model.aspect_encoder.coef_)
        _ = model.kp_encoder.weight.data.copy_(np_asp_weight.data)

        ############################
        # KP projection

        # |K| x E
        new_kpp_weight = torch.FloatTensor(
            np.vstack([estimator.coef_ for estimator in np_model._kp_proj.estimators_])
        )
        # |K| x 1 -> |K|
        new_kpp_bias = torch.FloatTensor(
            np.vstack(
                [estimator.intercept_ for estimator in np_model._kp_proj.estimators_]
            )
        ).squeeze()

        # Assign trained numpy weights
        _ = model.kp_proj.weight.data.copy_(new_kpp_weight.data)
        _ = model.kp_proj.bias.data.copy_(new_kpp_bias.data)

        return model

    def forward(self, u, i, j, target_kps, user_kps, *args, **kwargs):
        # User latent factors - B x Na -> B x K
        latent_u = self.encode_user(u, user_kps)

        # Item latent factors - B -> B x K
        latent_i = self.gamma_i(i)
        latent_j = self.gamma_i(j)

        # Latent factor scores - B
        x_ui = torch.mul(latent_u, latent_i).sum(dim=-1)
        x_uj = torch.mul(latent_u, latent_j).sum(dim=-1)

        kps_ui, kp_loss = None, None
        if self.kp_proj is not None:
            # Predict kps - B x K -> B x A
            kps_ui = self.kp_proj(latent_u + latent_i)
            # BCE loss against B x A boolean target
            kp_loss = F.binary_cross_entropy_with_logits(
                input=kps_ui, target=target_kps.float()
            )

        # Compute loss
        bpr_loss = -F.logsigmoid(x_ui - x_uj).mean()
        return x_ui, x_uj, kps_ui, bpr_loss, kp_loss

    def encode_user(self, u, user_kps, user_embedding: torch.FloatTensor = None):
        # User latent factors - B x Na -> B x K
        latent_u = self.kp_encoder(user_kps)
        if self.use_user_emb:
            if user_embedding is None:
                user_embedding = self.gamma_u(u)
            latent_u = latent_u + user_embedding
        return latent_u, user_embedding

    @classmethod
    def downweight_neg_items(
        cls, scores, neg_critiques, penalty: int = CRITIQUE_WEIGHTING
    ):
        if neg_critiques is not None:
            for ix, u_neg_crit in enumerate(neg_critiques):
                scores[ix, u_neg_crit] = scores[ix, u_neg_crit] - penalty
        return scores

    @classmethod
    def upweight_pos_items(
        cls, scores, pos_critiques, penalty: int = CRITIQUE_WEIGHTING
    ):
        if pos_critiques is not None:
            for ix, u_pos_crit in enumerate(pos_critiques):
                scores[ix, u_pos_crit] = scores[ix, u_pos_crit] + penalty
        return scores

    @classmethod
    def filter_invalid(cls, scores, invalid):
        # Remove invalid items (list of tensor indices)
        batch_size = scores.shape[0]
        n_items = scores.shape[-1]
        n_candidates = [n_items] * batch_size
        if invalid is not None:
            for ix, u_inv in enumerate(invalid):
                scores[ix, u_inv] = -1e5
                n_candidates[ix] -= len(u_inv)

        return scores, n_candidates

    def get_scores(
        self,
        u,
        user_kps,
        latent_u=None,
        user_embedding=None,
        invalid=None,
        neg_critiques=None,
        return_latent: bool = False,
        return_embedding: bool = False,
        *args,
        **kwargs
    ):
        # B x K
        if latent_u is None:
            latent_u, user_embedding = self.encode_user(
                u, user_kps, user_embedding=user_embedding
            )

        # Item latent * User latent -> B x I
        item_scores = torch.matmul(latent_u, self.gamma_i.weight.T)

        # Downrank filtered items
        item_scores = self.downweight_neg_items(item_scores, neg_critiques)

        # Remove invalid items (list of tensor indices)
        item_scores, n_candidates = self.filter_invalid(item_scores, invalid)

        # # Softmax
        # item_scores = F.softmax(item_scores, dim=-1)

        returns = [item_scores, n_candidates]

        if return_latent:
            returns.append(latent_u)

        if return_embedding:
            returns.append(user_embedding)

        return returns

    def get_ranks(
        self, u, user_kps, item_scores=None, index_by_item: bool = True, *args, **kwargs
    ):
        # B x I
        if item_scores is None:
            item_scores, *_ = self.get_scores(u, user_kps, *args, **kwargs)

        # item_ranks[b, i] = i-ranked item
        item_ranks = torch.argsort(item_scores, dim=-1, descending=True)
        if index_by_item:
            # item_ranks[b, i] = rank of item i
            item_ranks = torch.argsort(item_ranks, dim=-1, descending=False)

        return item_ranks

    def get_kps(
        self,
        u,
        i,
        user_kps,
        latent_u=None,
        latent_i=None,
        user_embedding=None,
        threshold: float = None,
        sample: str = "bernoulli",
        *args,
        **kwargs
    ):
        # B x K
        if latent_u is None:
            latent_u, user_embedding = self.encode_user(
                u, user_kps, user_embedding=user_embedding
            )
        if latent_i is None:
            latent_i = self.gamma_i(i)

        # Scores
        x_ui = torch.mul(latent_u, latent_i).sum(dim=-1)

        # Predict KPs - B x K -> B x A
        kp_logits = None
        if self.kp_proj is not None:
            kp_logits = self.kp_proj(latent_u + latent_i)

        # Predict binary KPs - B x A 0/1
        if sample == "bernoulli":
            kp_bin = torch.bernoulli(torch.sigmoid(kp_logits))
        elif threshold:
            kp_bin = torch.sigmoid(kp_logits) > threshold
        else:
            raise NotImplementedError(
                "Must specify sampling distribution or threshold to generate justifications!"
            )

        return x_ui, kp_logits, kp_bin, latent_u, latent_i
