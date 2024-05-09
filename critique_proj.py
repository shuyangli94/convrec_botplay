# Load model
import copy
import os
import random
from collections import Counter
from datetime import datetime, timedelta
from itertools import product

import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange

from convrec.data import UIJKPDataModule
from convrec.loading import load_model
from convrec.utils import count_parameters, df_to_csv, get_last_mod_ckpt, load_config
from critique_utils import (
    basic_stats,
    build_critique,
    build_turn_df,
    compressed_aspects,
    get_gold_rank,
    get_invalid_from_critiques,
    torch_fillna,
    turn_df_stats,
    vector_aspects,
)


def make_recommendation(
    u_input: torch.LongTensor,
    target_aspects: torch.BoolTensor,
    invalid: list,
    downweight_items: list,
    critique_vector: torch.FloatTensor,
    model: nn.Module,
    gm_scale: float = 0.0,
    window: int = 1,
    kp_threshold: float = None,
    kp_sample: str = "bernoulli",
    orig_scores: torch.FloatTensor = None,
):
    # Scores: B x Ni
    scores, n_cand, latent_u_ret = model.get_scores(
        u=u_input,
        user_kps=critique_vector,
        invalid=invalid,
        neg_critiques=downweight_items,
        return_latent=True,
    )

    # Post-norm fusion if necessary '
    # (GM scaling, don't bother on first turn w no critique)
    # 1.0 GM scaling means don't scale at all
    if gm_scale and gm_scale != 1.0 and orig_scores is not None:
        scores = (torch.softmax(scores, dim=-1) ** gm_scale) * (
            torch.softmax(orig_scores, dim=-1) ** (1 - gm_scale)
        )

    # Recommend items (B x W)
    topk = torch.topk(scores, k=window, dim=-1)
    rec_scores = topk.values  # B x W
    rec_items = topk.indices  # B x W

    #############################################################################
    # Aspect-based ranking of items

    # Latent Is -> B x W x H
    latent_i = model.gamma_i.weight[rec_items]

    # Get aspects -> B x W x Na
    # x_ui, kp_logits, kp_bin, latent_u, latent_i
    _, aspect_logits, aspects_bin, _, _ = model.get_kps(
        u=u_input,
        user_kps=critique_vector,
        i=rec_items,
        latent_u=latent_u_ret,
        latent_i=latent_i,
        threshold=kp_threshold,
        sample=kp_sample,
    )

    # Compute precision/recall/F1 between everything
    # B x W
    n_match = (aspects_bin.bool() & target_aspects.bool()).sum(dim=-1).float()
    n_pred = aspects_bin.sum(dim=-1).float()  # B x W
    n_gold = target_aspects.sum(dim=-1).float()  # W
    p_s = torch_fillna(n_match / n_pred, 0.0)  # B x W
    r_s = torch_fillna(n_match / n_gold, 0.0)  # B x W
    f1_s = torch_fillna(2.0 * p_s * r_s / (p_s + r_s), 0.0)  # B x W

    # Returns in order
    outputs = [
        scores,  # B x Ni, float
        n_cand,  # B, long
        rec_items,  # B x W, long
        rec_scores,  # B x W, float
        aspect_logits,  # B x W x Na, float
        aspects_bin,  # B x W x Na, bool
        p_s,  # B x W, float
        r_s,  # B x W, float
        f1_s,  # B x W, float
    ]

    return outputs


"""
Critique using recommender models (BPR)
"""
if __name__ == "__main__":
    import json
    import argparse
    from convrec.utils import _load, _save

    parser = argparse.ArgumentParser()

    # Critiquing hyperparams
    parser.add_argument("--gm-scale", type=str, default="0.85")

    # Affecting aspect/justification generation
    parser.add_argument("--kp-gen", type=str, default="bernoulli")

    # Affecting building the critique
    parser.add_argument("--user-strat", type=str, default="random")
    parser.add_argument("--allow-repeat", type=str, default="True")
    parser.add_argument("--aspect-filter", type=str, default="False")
    parser.add_argument("--feedback-type", type=str, default="N")  # N/P/NP
    parser.add_argument(
        "--n-feedback", type=str, default="1"
    )  # Number of pieces of feedback
    parser.add_argument("--criterion", type=str, default="f1")
    parser.add_argument("--target", type=str, default="item")  # Item/review as target
    parser.add_argument("--allow-arbitrary", action="store_true", default=False)

    # Logging and logistics
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--window", type=str, default="1")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max-session-length", type=int, default=10)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    start_script = datetime.now()

    print("Using arguments:")
    print(json.dumps(vars(args), indent=2, default=str))

    # Load the config & the right items etc.
    model_dir = args.model_dir
    model_name = os.path.basename(model_dir)

    # Where to save critiquing outputs
    output_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "Saving critique outputs to {} ({:,} existing outputs)".format(
            output_dir, len(os.listdir(output_dir))
        )
    )

    cfg_loc = os.path.join(model_dir, "config.yaml")
    cfg = load_config(cfg_loc)
    if not os.path.exists(os.path.join(model_dir, "checkpoints")):
        ckpt = os.path.join(model_dir, "last.ckpt")
    else:
        ckpt = get_last_mod_ckpt(
            os.path.join(model_dir, "checkpoints"), allow_last=False
        )

    # CONFIG DEFAULTS
    for attr, default in [
        ("loss", "CE"),  # fine-tuning loss
        ("fb_type", "N"),  # fine-tuning feedback
        ("behavior", "coop"),  # fine-tuning seeker model
    ]:
        if getattr(cfg, attr, None) is None:
            setattr(cfg, attr, default)
            print("Set default attribute {}: {}".format(attr, getattr(cfg, attr)))

    # Prepare data
    dm = UIJKPDataModule(
        splits_loc=cfg.splits_loc,
        kp_loc=cfg.kp_loc,
        batch_size=32,
        workers=0,
        neg_subset_ratio=None,
        shuffle_train=True,
        use_user_kp=True,
        pin_memory=False,
    )

    # Validation or testing dataset
    dm.setup("fit" if args.split == "valid" else "test")

    # Keyphrase popularity for critique building
    KP_POPULARITY = Counter()
    for i, kpp in dm.item_kp_train.items():
        KP_POPULARITY.update(kpp)

    # Modify config with the proper sizings
    cfg.n_kp = len(dm.kp_map)
    cfg.n_items = len(dm.item_map)
    cfg.n_users = len(dm.user_map)

    # Maximum session lengths
    if args.max_session_length > 0:
        MAX_SESSION = args.max_session_length
    else:
        MAX_SESSION = len(dm.item_map)
    print("Maximum {:,} turns per session".format(MAX_SESSION))

    # Actually load model and set evaluation mode
    config, model = load_model(ckpt_path=ckpt, cfg=cfg, strict=True)
    model.eval()
    # Block gradient computation across parameters
    for p in model.parameters():
        p.requires_grad = False
    print("Loaded model with {:,} params".format(count_parameters(model)))

    # Examples from evaluation dataset
    dataset_dir = os.path.dirname(cfg.splits_loc)
    eval_ix_loc = os.path.join(
        dataset_dir,
        "{}{}{}.pkl".format(
            args.split,
            "-S" if args.shuffle else "",
            f"-{args.sample}" if args.sample else "",
        ),
    )
    if os.path.exists(eval_ix_loc):
        key_indices = _load(eval_ix_loc, "Evaluation keys")
    else:
        indices = list(range(len(dm.eval_dataset)))
        if args.shuffle:
            random.Random(2021).shuffle(indices)
        if args.sample:
            indices = indices[: args.sample]
        key_indices = [dm.eval_dataset.index[ix] for ix in indices]
        _save(key_indices, eval_ix_loc, "{:,} evaluation keys".format(len(key_indices)))
    print("Evaluating over {:,} U-I examples".format(len(key_indices)))

    # GRID SEARCH
    # Parameters (grid search)
    hparams = []
    hp_labels = []
    hp_labels_short = []
    n_grid = 1
    for hp_arg, conv_fxn, lab, lab_short in [
        # Geometric mean scaling
        (args.gm_scale, float, "gm_scale (geometric mean weight)", "gm_scale"),
        # Window size (prev: Wdw)
        (args.window, int, "Window size", "Wdw"),
        # Aspect filter (prev: AF)
        (
            args.aspect_filter,
            lambda x: x.lower().strip() == "true",
            "Aspect filter",
            "filter",
        ),
        # User strategy (prev: Usr)
        (args.user_strat, str, "User strategy", "behavior"),
        # Allow repeat critiquing (prev: Rep)
        (
            args.allow_repeat,
            lambda x: x.lower().strip() == "true",
            "Allow repeat critiquing",
            "repeat",
        ),
        # Types of feedback to give N/P/NP
        (args.feedback_type, str, "Feedback type", "fb"),
        # Number of each type of feedback to give
        (args.n_feedback, int, "Number of pieces of feedback", "n_fb"),
        # Criterion for selecting
        (args.criterion, str, "Criterion for selecting from window", "win_crit"),
        # Criterion for generating aspects
        (args.kp_gen, str, "Justification generation method", "just_gen"),
        # Use gold review OR gold item historical aspects as target
        (args.target, str, "Aspect target", "kp_tgt"),
    ]:
        hp_candidates = [conv_fxn(val) for val in hp_arg.split(",")]
        hparams.append(hp_candidates)
        hp_labels.append(lab)
        hp_labels_short.append(lab_short)
        n_grid *= len(hp_candidates)
        print(
            '{:,} candidates for "{}": {}'.format(
                len(hp_candidates), lab, hp_candidates
            )
        )

    print("\n\n---------------------------------------------------")
    print("Grid searching over {:,} configurations".format(n_grid))

    grid_stats = []
    for ixx, grid_hp in enumerate(product(*hparams)):
        # Logging
        print(f"\n=== Grid {ixx}/{n_grid} ({ixx*100.0/n_grid:.2f}%) ============")
        for lab, hp_val in zip(hp_labels, grid_hp):
            print(f"{lab} := {hp_val}")
        if ixx > 0:
            s_elapsed = (datetime.now() - start_script).total_seconds()
            p_done = ixx / n_grid
            s_left = s_elapsed / p_done - s_elapsed
            print(
                "{} to finish {:.2f}% - estimated {} left".format(
                    timedelta(seconds=s_elapsed),
                    p_done * 100.0,
                    timedelta(seconds=s_left),
                )
            )

        print("----\n")

        # Extract hyperparams
        (
            gm_scale,
            window,
            aspect_filter,
            user_strat,
            allow_repeat,
            feedback_type,
            n_fb,
            rec_criterion,
            just_gen,
            kp_target,
        ) = grid_hp
        # No updating will occur
        if gm_scale == 0.0:
            continue
        crit_hparams = dict(zip(hp_labels_short, grid_hp))

        # Parsing justification generation
        if just_gen == "bernoulli":
            kp_sample = "bernoulli"
            kp_threshold = None
        elif just_gen.startswith("t"):
            kp_sample = None
            kp_threshold = float(just_gen[1:])
        else:
            raise NotImplementedError(
                f'Justification generation style "{just_gen}" not supported'
            )

        # Wipe gradients, initialize
        model.zero_grad()

        # where to store critiques
        exp_name = "PROJ_gm{gm_scale}{wdw}{kptgt}{kp_filter}{strat}{rep}{fb}{nfb}{rcrit}{just}{samp}{split}".format(
            gm_scale=gm_scale,  # Geometric mean
            wdw=f"-W{window}" if window > 1 else "",  # Window size (# recs)
            kptgt=f"-T{kp_target}",  # Item or review as gold target for critiques
            kp_filter=f"-AF" if aspect_filter else "",  # Aspect filter
            strat=f"-{user_strat}"
            if user_strat != "random"
            else "",  # User behavior (default: random)
            rep=f"-noR"
            if not allow_repeat
            else "",  # Allow repeat critiquing (default: true)
            fb=f"-fb{feedback_type}" if feedback_type != "N" else "",
            nfb=f"-nfb{n_fb}" if n_fb != 1 else "",
            rcrit=f"-C{rec_criterion}" if rec_criterion != "f1" else "",
            just=f"-{just_gen}" if just_gen != "bernoulli" else "",
            samp="-samp" + str(args.sample) if args.sample else "",
            split=args.split if args.split != "valid" else "",
        )
        exp_loc = os.path.join(output_dir, f"{exp_name}.pkl")
        if not args.overwrite and os.path.exists(exp_loc):
            loaded_crit_hparams, exp_record = _load(
                exp_loc, "existing experiment records"
            )
            if not all(loaded_crit_hparams[k] == v for k, v in crit_hparams.items()):
                print("==== LOADED HPARAMS DO NOT MATCH ====")
                print("Set: {}".format(crit_hparams))
                print("Loaded: {}".format(loaded_crit_hparams))
                exp_record = dict()
            else:
                print("Using {:,} previously saved records".format(len(exp_record)))
        else:
            exp_record = dict()
            print("No previous records of this experiment exist")
        crit_hparams["exp"] = exp_name
        crit_hparams["model"] = model_name

        # Track stats for all examples
        stats = []
        checkpoint_frequency = max(50, len(key_indices) // 5)
        max_items = len(key_indices)
        for pos, key in enumerate(key_indices):
            # SKIP ALREADY EVALUATED
            if key in exp_record:
                continue

            # u, i, j, aspects, i_kp, j_kp
            u, gold, _, review_aspects, _, _, u_kp = dm.eval_dataset._get_item_by_key(
                key
            )

            # Priors
            u_input = torch.LongTensor([u])
            i_input = torch.LongTensor([gold])
            if kp_target == "review":
                target_bool = torch.BoolTensor(review_aspects)
                compressed_targets = compressed_aspects(review_aspects)
            elif kp_target == "item":
                compressed_targets = list(dm.item_kp_train[gold].keys())
                target_bool = torch.BoolTensor(
                    vector_aspects(compressed_targets, n_aspects=len(dm.kp_map))
                ).unsqueeze(0)
            invalid_items = list(dm.train_ui[u])
            prev_neg_critiques = set()
            prev_pos_critiques = set()
            orig_scores = None
            allowed_aspects = set(range(len(dm.kp_map)))

            # Initialize with user training aspects into B x V vector
            critique_vector = torch.FloatTensor([u_kp])

            # Could result in invalid results
            if gold in invalid_items:
                continue

            # Tracking
            tracked_turns = []
            n_critiques = 0

            # FOR EACH TURN OF DIALOG
            gold_is_returned = False
            pb = trange(MAX_SESSION, desc="placeholder", leave=True)
            for n_iter in pb:
                turn_start = datetime.now()
                if aspect_filter:
                    # FILTER INVALID ITEMS
                    invalid_items = list(
                        set(invalid_items)
                        | get_invalid_from_critiques(
                            item_kps=dm.item_kp_train,
                            neg=prev_neg_critiques,
                            pos=prev_pos_critiques,
                        )
                    )

                # Get scores, recommendation, latents
                with torch.no_grad():
                    (
                        curr_scores,
                        curr_n_cand,
                        curr_rec_items,
                        curr_rec_scores,
                        curr_kp_logits,
                        curr_rec_kps,
                        curr_rec_p,
                        curr_rec_r,
                        curr_rec_f1,
                    ) = make_recommendation(
                        u_input=u_input,
                        target_aspects=target_bool,
                        invalid=[invalid_items],
                        downweight_items=None,
                        critique_vector=critique_vector,
                        model=model,
                        gm_scale=gm_scale,
                        window=window,
                        kp_threshold=kp_threshold,
                        kp_sample=kp_sample,
                        orig_scores=orig_scores,
                    )

                # Original scores so we can cache it in case gm_scale > 0
                if orig_scores is None:
                    orig_scores = curr_scores

                # Get gold rank - scalar
                curr_gold_rank = get_gold_rank(curr_scores, gold)

                # Look at all recommended items, pick out gold if necessary
                turn_recs = curr_rec_items.squeeze(0).cpu().numpy().tolist()

                # Pick top recommended item from window - (,)
                if gold in turn_recs:
                    gold_is_returned = True
                    top_rec_ix = turn_recs.index(gold)
                # Pick top item by F1 with target
                elif rec_criterion == "f1":
                    top_rec_ix = torch.argmax(curr_rec_f1, dim=-1).item()
                # Pick first item and justification
                elif rec_criterion == "first":
                    top_rec_ix = 0
                else:
                    raise NotImplementedError(
                        f'Unsupported intra-window selection strategy "{rec_criterion}" specified'
                    )

                curr_item = curr_rec_items[0, top_rec_ix].item()  # (,)
                curr_kp_bin = curr_rec_kps[0, top_rec_ix]  # B x K

                # Tracking
                duration = (datetime.now() - turn_start).total_seconds()
                turn_start = datetime.now()
                orig_gr = (
                    tracked_turns[0]["gold_rank"]
                    if tracked_turns
                    else curr_gold_rank.item()
                )
                orig_turns = orig_gr // window + (1 if orig_gr % window else 0)
                orig_gr_at_turn = max(0, orig_gr - window * n_iter)
                tracked_turns.append(
                    {
                        "time": duration,  # Seconds
                        "base_gold_rank": orig_gr_at_turn,  # Rank of gold item if no actions were taken
                        "gold_rank": curr_gold_rank.item(),  # Rank of gold item @ turn
                        "gold_aspects": compressed_targets,  # Gold aspects
                        "n_cand": curr_n_cand[  # Number of remaining items to choose from
                            0
                        ],
                        "recs": turn_recs,  # All recommended items
                        "top_rec": curr_item,  # Most closely matching aspects predicted
                        "rec_p": curr_rec_p.squeeze(0)  # Aspect precision for all recs
                        .cpu()
                        .numpy()
                        .tolist(),
                        "rec_r": curr_rec_r.squeeze(0)  # Aspect recall for all recs
                        .cpu()
                        .numpy()
                        .tolist(),
                        "rec_f1": curr_rec_f1.squeeze(0)  # Aspect F1 for all recs
                        .cpu()
                        .numpy()
                        .tolist(),
                        "rec_kps": [  # Binary aspects for all recs, compressed to indices
                            compressed_aspects(k)
                            for k in curr_rec_kps.squeeze(0).cpu().numpy().tolist()
                        ],
                        "kp_logits": curr_kp_logits,  # Logits for all KPs, B x W x Na
                        "item_scores": curr_scores,  # Scores of all items, B x Ni
                    }
                )

                # Update progress bar with gold rank/current turn
                pb.set_description(
                    "[{}/{}] ({},{}) {}->{} ({}->{}) vs. {} ({} crit)".format(
                        pos + 1,
                        max_items,
                        u,
                        gold,
                        orig_gr,
                        curr_gold_rank.item(),
                        orig_turns,
                        n_iter,
                        orig_gr_at_turn,
                        n_critiques,
                    ),
                    refresh=True,
                )

                # Update visited set
                invalid_items.extend(turn_recs)

                # Gold in top K OR reached maximum session length
                if gold_is_returned or n_iter >= MAX_SESSION:
                    break

                # Build critique based on what was shown to the user
                curr_kp_list = curr_kp_bin.squeeze().cpu().numpy().tolist()
                pos_crit, neg_crit = build_critique(
                    target=compressed_targets,
                    prediction=compressed_aspects(curr_kp_list),
                    behavior=user_strat,
                    prev_neg_critiques=prev_neg_critiques,
                    prev_pos_critiques=prev_pos_critiques,
                    allowed_aspects=allowed_aspects,
                    allow_repeat=allow_repeat,
                    kp_popularity=KP_POPULARITY,
                    fb_type=feedback_type,
                    cand_aspect_freq=dm.item_kp_train[int(curr_item)],
                    target_aspect_freq=dm.item_kp_train[gold],
                    n_fb=n_fb,
                    allow_arbitrary_critique=args.allow_arbitrary,
                )

                # Modify critique vector
                for neg_k in neg_crit:
                    penalty = dm.user_kp_train[u].get(neg_k, 1)
                    critique_vector[0, neg_k] -= penalty
                    n_critiques += 1
                for pos_k in pos_crit:
                    gain = dm.user_kp_train[u].get(pos_k, 1)
                    critique_vector[0, pos_k] += gain
                    n_critiques += 1

                # Update set of critiqued aspects to downweight/filter items
                prev_neg_critiques |= set(neg_crit)
                prev_pos_critiques |= set(pos_crit)

            # Record experiment
            exp_record[key] = tracked_turns

            # CHECKPOINT RECORD
            if pos % checkpoint_frequency == 0:
                _save(
                    (crit_hparams, exp_record),
                    exp_loc,
                    "{:,} checkpointed experiment records".format(len(exp_record)),
                )

        # Save all records
        _save(
            (crit_hparams, exp_record),
            exp_loc,
            "all {:,} records of this exp".format(len(exp_record)),
        )

        # Compute stats & print, store to grid
        exp_stats = copy.deepcopy(crit_hparams)
        exp_stats.update(basic_stats(exp_record))
        turn_df = build_turn_df(exp_record, window_size=window)
        turn_stats, *_ = turn_df_stats(turn_df, window_size=window)
        exp_stats.update(turn_stats)
        grid_stats.append(exp_stats)

        # Accumulate stats
        df = pd.DataFrame(grid_stats)
        # Re-organize the columns & sort
        set_order_columns = [
            "filter",
            "fb",
            "behavior",
            "exp",
            "model",
            "max10_HR@1",
            "max10_HR@5",
            "max10_HR@10",
            "mean_c_turns",
            "med_o_turns",
            "med_c_turns",
        ]
        df = df[
            set_order_columns + [c for c in df.columns if c not in set_order_columns]
        ].sort_values(
            [
                "max10_HR@1",
                "max10_HR@5",
                "max10_HR@10",
            ],
            ascending=False,
        )

        print("\n------\n")
        print(df)
        print("------\n")
        df_to_csv(df)
        print("------\n")
        agg_save_loc = os.path.join(args.out_dir, f"{args.name}.pkl")
        _save(df, agg_save_loc, "{:,} experiment aggregate stat rows".format(len(df)))
