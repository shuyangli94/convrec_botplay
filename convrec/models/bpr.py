import torch
import torch.nn as nn
import torch.nn.functional as F

from convrec.utils import count_parameters

CRITIQUE_WEIGHTING = 100


class BPR(nn.Module):
    """
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf

    BPR(u, i, j) = Sum[ln(sig(x_ui - x_uj))] + lambda * L2(theta)
    x_ui = gamma_u(u) dot gamma_i(i) + beta_u(u) + beta_i(i) + alpha
    x_uj = gamma_u(u) dot gamma_i(j) + beta_u(u) + beta_i(j) + alpha
    """

    def __init__(self, k: int, n_items: int, n_users: int, beta: bool = True):
        super().__init__()

        # Params
        self.k = k
        self.n_items = n_items
        self.n_users = n_users
        self.has_beta = beta

        # User/item biases
        if self.has_beta:
            self.beta_i = nn.Embedding(self.n_items, 1)
            self.beta_u = nn.Embedding(self.n_users, 1)

        # Latent representations
        self.gamma_i = nn.Embedding(self.n_items, k)
        self.gamma_u = nn.Embedding(self.n_users, k)

        print(
            'Created {} for {:,} users, {:,} items, k={} with {:,} parameters'.
            format(self.__class__.__name__, self.n_users, self.n_items, self.k,
                   count_parameters(self)))

    def forward(self, u, i, j, *args, **kwargs):
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

        # Compute loss
        loss = -F.logsigmoid(x_ui - x_uj).mean()
        return x_ui, x_uj, loss

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
