import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from convrec.utils import count_parameters

CRITIQUE_WEIGHTING = 100


class BPRKP(nn.Module):
    """
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf

    BPR(u, i, j) = Sum[ln(sig(x_ui - x_uj))] + lambda * L2(theta)
    x_ui = gamma_u(u) dot gamma_i(i) + beta_u(u) + beta_i(i) + alpha
    x_uj = gamma_u(u) dot gamma_i(j) + beta_u(u) + beta_i(j) + alpha
    """

    def __init__(self,
                 k: int,
                 n_items: int,
                 n_users: int,
                 n_kp: int,
                 kp_layers: int = 0,
                 beta: bool = True):
        super().__init__()

        # Params
        self.k = k
        self.n_items = n_items
        self.n_users = n_users
        self.n_kp = n_kp
        self.kp_layers = kp_layers
        self.has_beta = beta

        # User/item biases
        if self.has_beta:
            self.beta_i = nn.Embedding(self.n_items, 1)
            self.beta_u = nn.Embedding(self.n_users, 1)

        # Latent representations
        self.gamma_i = nn.Embedding(self.n_items, k)
        self.gamma_u = nn.Embedding(self.n_users, k)

        # Aspect projection
        self.kp_proj = None
        if self.n_kp:
            if self.kp_layers > 0:
                # Hidden layers
                layers = []
                for _ in range(self.kp_layers):
                    layers.extend([nn.Linear(self.k, self.k), nn.ReLU()])
                # Final projection
                layers.append(nn.Linear(self.k, self.n_kp))
                self.kp_proj = nn.Sequential(*layers)
            else:
                self.kp_proj = nn.Linear(self.k, self.n_kp)

        print(
            'Created {} for {:,} users, {:,} items, k={} with {:,} parameters'.
            format(self.__class__.__name__, self.n_users, self.n_items, self.k,
                   count_parameters(self)))

    def forward(self, u, i, j, kps, *args, **kwargs):
        # Latent factors - B x K
        latent_u = self.gamma_u(u)
        latent_i = self.gamma_i(i)
        latent_j = self.gamma_i(j)

        # Latent factor scores - B
        x_ui = torch.mul(latent_u, latent_i).sum(dim=-1)
        x_uj = torch.mul(latent_u, latent_j).sum(dim=-1)

        # Bias terms - B x 1 -> B
        if self.has_beta:
            bias_u = self.beta_u(u).squeeze()
            bias_i = self.beta_i(i).squeeze()
            bias_j = self.beta_i(j).squeeze()

            x_ui = x_ui + bias_u + bias_i
            x_uj = x_uj + bias_u + bias_j

        kps_ui, kp_loss = None, None
        if self.kp_proj is not None:
            # Predict kps - B x K -> B x A
            kps_ui = self.kp_proj(latent_u + latent_i)
            # BCE loss against B x A boolean target
            kp_loss = F.binary_cross_entropy_with_logits(
                input=kps_ui, target=kps.float())

        # Compute loss
        bpr_loss = -F.logsigmoid(x_ui - x_uj).mean()
        return x_ui, x_uj, kps_ui, bpr_loss, kp_loss

    @classmethod
    def downweight_neg_items(cls,
                             scores,
                             neg_critiques,
                             penalty: int = CRITIQUE_WEIGHTING):
        if neg_critiques is not None:
            for ix, u_neg_crit in enumerate(neg_critiques):
                scores[ix, u_neg_crit] = scores[ix, u_neg_crit] - penalty
        return scores

    @classmethod
    def upweight_pos_items(cls,
                             scores,
                             pos_critiques,
                             penalty: int = CRITIQUE_WEIGHTING):
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
            latent_u = self.gamma_u(u)

        # Item latent * User latent -> B x I
        item_scores = torch.matmul(latent_u, self.gamma_i.weight.T)

        # We don't need to add user biases, but we need item biases
        if self.has_beta:
            item_scores = torch.add(item_scores, self.beta_i.weight.T)

        # Downrank filtered items
        item_scores = self.downweight_neg_items(item_scores, neg_critiques)

        # Remove invalid items (list of tensor indices)
        item_scores, n_candidates = self.filter_invalid(item_scores, invalid)

        if return_latent:
            return item_scores, n_candidates, latent_u

        return item_scores, n_candidates

    def get_ranks(self,
                  u,
                  item_scores=None,
                  index_by_item: bool = True,
                  *args,
                  **kwargs):
        # B x I
        if item_scores is None:
            item_scores, *_ = self.get_scores(u, *args, **kwargs)

        # item_ranks[b, i] = i-ranked item
        item_ranks = torch.argsort(item_scores, dim=-1, descending=True)
        if index_by_item:
            # item_ranks[b, i] = rank of item i
            item_ranks = torch.argsort(item_ranks, dim=-1, descending=False)

        return item_ranks

    def get_kps(self,
                u,
                i,
                latent_u=None,
                latent_i=None,
                threshold: float = None,
                sample: str = 'bernoulli',
                *args,
                **kwargs):
        # B x K
        if latent_u is None:
            latent_u = self.gamma_u(u)
        if latent_i is None:
            latent_i = self.gamma_i(i)

        # Scores
        x_ui = torch.mul(latent_u, latent_i).sum(dim=-1)
        # Bias terms - B x 1 -> B
        if self.has_beta:
            bias_u = self.beta_u(u).squeeze()
            bias_i = self.beta_i(i).squeeze()

            x_ui = x_ui + bias_u + bias_i

        # Predict KPs - B x K -> B x A
        kp_logits = None
        if self.kp_proj is not None:
            kp_logits = self.kp_proj(latent_u + latent_i)

        # Predict binary KPs - B x A 0/1
        if sample == 'bernoulli':
            kp_bin = torch.bernoulli(torch.sigmoid(kp_logits))
        elif threshold:
            kp_bin = torch.sigmoid(kp_logits) > threshold
        else:
            raise NotImplementedError(
                'Must specify sampling distribution or threshold to generate justifications!'
            )

        return x_ui, kp_logits, kp_bin, latent_u, latent_i
