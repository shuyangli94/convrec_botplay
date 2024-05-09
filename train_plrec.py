import os
from collections import Counter
import json

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

from convrec.data import UIJKPDataModule
from convrec.models import PLRecWithAspects, PLRecWithAspectsKP
from critique_utils import compressed_aspects


class PopRec(object):
    def __init__(self, train_ui: dict, n_items: int):
        self.pop_scores = Counter()
        for u, i_s in train_ui.items():
            for i in i_s:
                self.pop_scores[i] += 1
        self.pop_vec = [self.pop_scores[i] for i in range(n_items)]

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

    def get_scores(self, u, invalid=None, *args, **kwargs):
        batch_size = u.shape[0]
        # B x I
        scores = torch.FloatTensor(self.pop_vec).repeat(batch_size, 1)
        # B x I
        scores, n_candidates = self.filter_invalid(scores, invalid)
        return scores, n_candidates

    def get_ranks(
        self, u, item_scores=None, index_by_item: bool = True, *args, **kwargs
    ):
        # B x I
        if item_scores is None:
            item_scores, *_ = self.get_scores(u, *args, **kwargs)
        # item_ranks[b, i] = i-ranked item
        item_ranks = torch.argsort(item_scores, dim=-1, descending=True)
        if index_by_item:
            # item_ranks[b, i] = rank of item i
            item_ranks = torch.argsort(item_ranks, dim=-1, descending=False)
        return item_ranks


def get_model_outputs(model, dm):
    # Collect stats for a single model
    all_users = []
    all_gold_ranks = []
    all_n_cand = []
    all_target_kps = []
    all_pred_kps = []
    all_pred_item_kps = []
    n_batches = len(dm.eval_dataloader)
    for batch_ix, batch in tqdm(enumerate(dm.eval_dataloader), total=n_batches):
        u, i, j, tgt_aspects, i_kp, j_kp, u_kp = batch
        batch_size = u.shape[0]
        # Get scores
        with torch.no_grad():
            item_scores, n_cands = model.get_scores(
                u=u, user_kps=u_kp, invalid=[dm.train_ui[int(user)] for user in u]
            )
        # Get gold ranks (B,)
        item_ranks = model.get_ranks(
            u=u, user_kps=u_kp, item_scores=item_scores, index_by_item=True
        )
        gold_ranks = item_ranks[torch.arange(batch_size), i]
        # Pick top ranked item (B,)
        rec_items = model.get_ranks(
            u=u, user_kps=u_kp, item_scores=item_scores, index_by_item=False
        )[:, 0]
        # PREDICT KPs
        if hasattr(model, "get_kps"):
            # x_ui, kp_logits, kp_bin, latent_u, latent_i
            with torch.no_grad():
                _, _, pred_kps, _, _ = model.get_kps(u=u, user_kps=u_kp, i=rec_items)
            pred_kps = [set(compressed_aspects(k)) for k in pred_kps]
        else:
            # Get the KPs of the predicted item
            pred_kps = [set(dm.item_kp_train[item.item()].keys()) for item in rec_items]
        # Accumulate
        if isinstance(gold_ranks, torch.Tensor):
            all_gold_ranks.extend(gold_ranks.numpy().tolist())
        else:
            all_gold_ranks.extend(gold_ranks.tolist())
        all_users.extend(u.detach().squeeze().numpy().tolist())
        all_n_cand.extend(n_cands)
        all_target_kps.extend([set(dm.item_kp_train[item.item()].keys()) for item in i])
        all_pred_item_kps.extend(
            [set(dm.item_kp_train[item.item()].keys()) for item in rec_items]
        )
        all_pred_kps.extend(pred_kps)
    return (
        all_users,
        all_gold_ranks,
        all_n_cand,
        all_target_kps,
        all_pred_kps,
        all_pred_item_kps,
    )


def get_stats(
    model_name,
    all_gold_ranks,
    all_n_cand,
    all_target_kps,
    all_pred_kps,
    all_pred_item_kps,
):
    # Calculate split statistics
    stat_dict = {"model": model_name}
    # AUC
    auc = np.mean([(nc - gr) / nc for nc, gr in zip(all_n_cand, all_gold_ranks)])
    stat_dict["AUC"] = auc
    print(f"AUC: {auc}")
    # HR
    for k in [1, 5, 10, 20]:
        hr_k = sum(gr < k for gr in all_gold_ranks) / len(all_gold_ranks)
        stat_dict[f"HR_{k}"] = hr_k
        print(f"HR@{k}: {hr_k}")
    # KP stats - target item
    precs = []
    recs = []
    f1s = []
    for hyp, ref in zip(all_pred_kps, all_target_kps):
        n_match = len(hyp & ref)
        n_hyp = len(hyp)
        n_ref = len(ref)
        p = n_match / n_hyp if n_hyp else 0
        r = n_match / n_ref if n_ref else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    print(
        "Target KP P {}, R {}, F1 {}".format(
            np.mean(precs), np.mean(recs), np.mean(f1s)
        )
    )
    stat_dict["gold_P"] = np.mean(precs)
    stat_dict["gold_R"] = np.mean(recs)
    stat_dict["gold_F1"] = np.mean(f1s)
    # KP stats - predicted item
    precs = []
    recs = []
    f1s = []
    for hyp, ref in zip(all_pred_kps, all_pred_item_kps):
        n_match = len(hyp & ref)
        n_hyp = len(hyp)
        n_ref = len(ref)
        p = n_match / n_hyp if n_hyp else 0
        r = n_match / n_ref if n_ref else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    print(
        "Pred Item KP P {}, R {}, F1 {}".format(
            np.mean(precs), np.mean(recs), np.mean(f1s)
        )
    )
    stat_dict["pred_P"] = np.mean(precs)
    stat_dict["pred_R"] = np.mean(recs)
    stat_dict["pred_F1"] = np.mean(f1s)
    # Number of KPs predicted
    stat_dict["n_kp"] = np.mean([len(k) for k in all_pred_kps])
    print("Avg. {:.3f} KP predicted/item".format(stat_dict["n_kp"]))
    return stat_dict


def get_stats_topk(model_name, all_users, all_gold_ranks):
    dff = pd.DataFrame({"u": all_users, "gr": all_gold_ranks})
    # Calculate split statistics
    stat_dict = {"model": model_name}
    # MRR
    mrr = np.mean((1.0 / (1 + dff.groupby(["u"])["gr"].min())))
    stat_dict["MRR"] = mrr
    print(f"MRR: {mrr}")
    # NDCG
    for k in [5, 10, 20]:
        dff["found"] = [0 if gr > k else 1 for gr in dff["gr"].values]
        u_found = dff.groupby(["u"])["found"].sum().to_dict()
        u_idcg = {
            u: sum([1 / np.log2(p + 2) for p in range(uf)]) for u, uf in u_found.items()
        }
        dff["dcg"] = [0 if gr > k else 1 / np.log2(gr + 2) for gr in dff["gr"].values]
        dcg_k = dff.groupby(["u"])["dcg"].sum().to_dict()
        ndcg_k = np.mean([u_dcg / (u_idcg[u] or 1.0) for u, u_dcg in dcg_k.items()])
        stat_dict[f"NDCG_{k}"] = ndcg_k
        print(f"NDCG@{k}: {ndcg_k}")
    return stat_dict


"""
Train Recommender System (BPR)
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train PLRec")
    parser.add_argument(
        "--split-file", type=str, required=True, help="File containing data splits"
    )
    parser.add_argument(
        "--kp-file", type=str, required=True, help="File containing key phrases"
    )
    parser.add_argument(
        "--ds", type=str, required=True, help='e.g. "beer" or "cdvinyl"'
    )
    parser.add_argument(
        "--model-dir", type=str, required=True, help="Directory to save the model"
    )
    args = parser.parse_args()

    print("\n===== {} =====".format(args.ds))
    dm = UIJKPDataModule(
        splits_loc=args.split_file,
        kp_loc=args.kp_file,
        batch_size=128,
        pin_memory=False,
        workers=0,
        neg_subset_ratio=None,
        shuffle_train=True,
        use_user_kp=True,
    )
    dm.setup(
        "test"
    )  # PLRec training doesn't use the dm.*_dset attributes, so no need to create them in setup.
    MODELS = [
        ("popularity", "PopRec"),
        ("PLRec", "PLRec"),
        ("PLRecKP", "PLRecKP"),
    ]
    agg_stats = []
    # Create training data
    train_mat = np.zeros((len(dm.user_map), len(dm.item_map)), dtype=float)
    for u, i_s in dm.train_ui.items():
        for i in i_s:
            train_mat[u, i] = 1
    keyphrase_freq_mat = np.zeros((len(dm.user_map), len(dm.kp_map)), dtype=float)
    for u, kp_s in dm.user_kp_train.items():
        for kp, freq in kp_s.items():
            keyphrase_freq_mat[u, kp] = freq
    for model_name, label in MODELS:
        if label in {s["label"] for s in agg_stats}:
            print("SKIPPING {} - ALREADY DONE".format(label))
            continue
        if model_name == "popularity":
            # Popularity
            model = PopRec(dm.train_ui, len(dm.item_map))
        elif model_name == "PLRec":
            # PLRec
            if args.ds == "beer":
                n_iter = 10
                lamb = 80
                dim = 50
            elif args.ds == "cdvinyl":
                n_iter = 10
                lamb = 1000
                dim = 400
            elif args.ds == "goodreads":
                n_iter = 10
                lamb = 80
                dim = 50
            else:
                raise NotImplementedError("What ds? {}".format(args.ds))
            model = PLRecWithAspects(n_iter=n_iter, lamb=lamb, dim=dim)
            model_dir = os.path.join(
                args.model_dir, f"{model_name}-{args.ds}_I{n_iter}_L{lamb}_d{dim}"
            )
            os.makedirs(model_dir, exist_ok=True)
            conf_loc = os.path.join(model_dir, "config.json")
            with open(conf_loc, "w+") as wf:
                config = {"n_iter": n_iter, "lamb": lamb, "dim": dim}
                json.dump(config, wf)
            print(
                "Saved config to {} ({:.2f} KB):\n{}".format(
                    conf_loc,
                    os.path.getsize(conf_loc) / 1024,
                    json.dumps(config, indent=2, default=str),
                )
            )
            model = model.train_plrec(train_mat=train_mat)
            model = model.train_aspect_encoder(train_u_kp_freq=keyphrase_freq_mat)
            weights_path = os.path.join(model_dir, "model.pt")
            model.save_weights(path=weights_path)
        elif model_name == "PLRecKP":
            # PLRec
            if args.ds == "beer":
                n_iter = 10
                lamb = 80
                dim = 50
            elif args.ds == "cdvinyl":
                n_iter = 10
                lamb = 1000
                dim = 400
            elif args.ds == "goodreads":
                n_iter = 10
                lamb = 80
                dim = 50
            else:
                raise NotImplementedError("What ds? {}".format(args.ds))
            model = PLRecWithAspectsKP(
                n_kp=len(dm.kp_map), n_iter=n_iter, lamb=lamb, dim=dim
            )
            model_dir = os.path.join(
                args.model_dir, f"{model_name}-{args.ds}_I{n_iter}_L{lamb}_d{dim}"
            )
            os.makedirs(model_dir, exist_ok=True)
            conf_loc = os.path.join(model_dir, "config.json")
            with open(conf_loc, "w+") as wf:
                config = {
                    "n_iter": n_iter,
                    "lamb": lamb,
                    "dim": dim,
                    "n_kp": len(dm.kp_map),
                }
                json.dump(config, wf)
            print(
                "Saved config to {} ({:.2f} KB):\n{}".format(
                    conf_loc,
                    os.path.getsize(conf_loc) / 1024,
                    json.dumps(config, indent=2, default=str),
                )
            )
            # Train user/item embeddings
            model = model.train_plrec(train_mat=train_mat)
            # Train aspect encoder
            model = model.train_aspect_encoder(train_u_kp_freq=keyphrase_freq_mat)
            # Training data for KP projection
            # Create boolean of which KPs appear in training for each item
            item_kp_freq_mat = np.zeros((len(dm.item_map), len(dm.kp_map)), dtype=float)
            for i, kp_s in dm.item_kp_train.items():
                for kp, freq in kp_s.items():
                    item_kp_freq_mat[i, kp] = 1
            x_arr, y_arr = model.prepare_kp_train(
                train_ui=dm.train_ui, item_kp_freq_mat=item_kp_freq_mat
            )
            model = model.train_kp_proj(
                train_inputs=x_arr, train_targets=y_arr, max_iter=5000
            )
            # Seriailze
            weights_path = os.path.join(model_dir, "model.pt")
            model.save_weights(path=weights_path)
