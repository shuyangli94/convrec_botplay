import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from convrec.utils import count_parameters

CRITIQUE_WEIGHTING = 100

class BPRItemModel(nn.Module):
    """
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf

    BPR(u, i, j) = Sum[ln(sig(x_ui - x_uj))] + lambda * L2(theta)
    gamma_i = kp_proj.T * aspects_i
    x_ui = gamma_u(u) dot gamma_i(i)
    aspects_ui = kp_proj(x_ui + gamma_i)
    """

    def __init__(self,
                 k: int,
                 n_users: int,
                 n_items: int,
                 n_kp: int,
                 use_item_emb: bool = False):
        super().__init__()

        # Params
        self.k = k
        self.n_users = n_users
        self.n_items = n_items
        self.n_kp = n_kp
        self.use_item_emb = use_item_emb

        # Latent representations
        self.gamma_u = nn.Embedding(self.n_users, k)
        self.gamma_i = None
        if self.use_item_emb:
            self.gamma_i = nn.Embedding(self.n_items, k)
            print('Creating hybrid model with aspect encoder and item latent emb.')

        # Aspect projection
        self.kp_proj = nn.Linear(self.k, self.n_kp)

        print(
            'Created {} for {:,} users, {:,} aspects, k={} with {:,} parameters'.
            format(self.__class__.__name__, self.n_users, self.n_kp, self.k,
                   count_parameters(self)))

    def forward(self, u, i, j, i_kps, j_kps, kps):
        # Latent factors - B x K
        latent_u = self.gamma_u(u)

        # Item representations - kp_proj.T * aspects_i/j
        latent_i = torch.matmul(self.kp_proj.weight.T, i_kps.float().T).T
        latent_j = torch.matmul(self.kp_proj.weight.T, j_kps.float().T).T

        # If also using learned item embeddings
        if self.use_item_emb:
            emb_i = self.gamma_i(i)
            latent_i = latent_i + emb_i
            emb_j = self.gamma_i(j)
            latent_j = latent_j + emb_j

        # Latent factor scores - B x Na -> B x k
        x_ui = torch.mul(latent_u, latent_i).sum(dim=-1)
        x_uj = torch.mul(latent_u, latent_j).sum(dim=-1)

        # Predict kps - B x K -> B x A
        kps_ui = self.kp_proj(latent_u + latent_i)
        # BCE loss against B x A boolean target
        kp_loss = F.binary_cross_entropy_with_logits(
            input=kps_ui, target=kps.float())

        # Compute loss
        bpr_loss = -F.logsigmoid(x_ui - x_uj).mean()
        return x_ui, x_uj, kps_ui, bpr_loss, kp_loss

    def get_scores(self, u, all_item_kp, latent_u=None, invalid=None):
        # B x K
        if latent_u is None:
            latent_u = self.gamma_u(u)

        # Item latents via kp_proj.T -> Ni x k
        item_latent_all = torch.matmul(self.kp_proj.weight.T, all_item_kp.T).T

        # If also using item embeddings
        if self.use_item_emb:
            # Also Ni x k
            item_latent_all = item_latent_all + self.gamma_i.weight

        # B x k * k x Ni -> B x Ni
        # score_u,i = latent_u[u, :] dot item_latent_all[i, :]
        item_scores = torch.matmul(latent_u, item_latent_all.T)

        # Remove invalid items (list of tensor indices)
        n_candidates = [item_scores.shape[-1]] * len(u)
        if invalid is not None:
            for ix, u_inv in enumerate(invalid):
                item_scores[ix, u_inv] = -np.inf
                n_candidates[ix] -= len(u_inv)

        return item_scores, n_candidates

    def get_ranks(self,
                  u,
                  all_item_kp,
                  item_scores=None,
                  index_by_item: bool = True):
        # B x I
        if item_scores is None:
            item_scores = self.get_scores(u, all_item_kp)

        # item_ranks[b, i] = i-ranked item
        item_ranks = torch.argsort(item_scores, dim=-1, descending=True)
        if index_by_item:
            # item_ranks[b, i] = rank of item i
            item_ranks = torch.argsort(item_ranks, dim=-1, descending=False)

        return item_ranks

    def get_kps(self, u, i):
        # B x K
        latent_u = self.gamma_u(u)
        latent_i = torch.matmul(self.kp_proj.weight.T, i.float().T).T

        # B x k
        if self.use_item_emb:
            emb_i = self.gamma_i(i)
            latent_i = latent_i + emb_i

        # Scores
        x_ui = torch.mul(latent_u, latent_i).sum(dim=-1)

        # Predict KPs - B x K -> B x A
        kp_logits = None
        if self.kp_proj is not None:
            kp_logits = self.kp_proj(latent_u + latent_i)

        return x_ui, kp_logits
