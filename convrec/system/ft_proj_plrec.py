import json
import os

import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from convrec.models import PLRecWithAspectsKPTorch
from convrec.optimizer import Adam, Lamb, RAdam
from convrec.utils import (
    check_nan,
    count_parameters,
    debug_log,
    get_last_mod_ckpt,
    load_checkpoint,
    load_config,
)


class System(pl.LightningModule):
    saved_metrics = [
        "global_step",
        "epoch",
        "loss",
        "mrr@1",
        "mrr@5",
        "mrr@10",
        "hr@1",
        "hr@5",
        "hr@10",
        "sr@1",
        "sr@5",
        "sr@10",
        "tt1",
        "tt5",
        "tt10",
    ]
    choice_metric = "loss"

    def __init__(
        self, config, train_ui: dict = None, kp_pop_order: list = None, **kwargs
    ):
        super().__init__()

        if kwargs:
            print(">>> IGNORING KWARGS:")
            print(json.dumps(kwargs, indent=2, default=str))

        # Store configurations
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.debug_mode = self.config.debug
        if self.debug_mode:
            print("~~~ DEBUG MODE ~~~")

        # Metrics
        self.metrics_loc = os.path.join(
            self.config.ckpt_path, self.config.name, "metrics.csv"
        )
        print("\n>> Saving metrics to {}".format(self.metrics_loc))

        # Training UI if being used to compute metrics
        self.train_ui = train_ui
        if self.train_ui:
            print(">> Evaluating metrics (HR, MRR, etc.) by removing training items")
        self.kp_pop_order = kp_pop_order
        if self.kp_pop_order:
            print(
                ">> User model picks most popular aspect from {:,}".format(
                    len(self.kp_pop_order)
                )
            )

        # Training parameters - recommendation loss
        self.loss_type = getattr(self.config, "loss", "CE")
        if self.loss_type == "CE":
            print(">> Fine-tuning with Cross-Entropy loss (softmax)")
        elif self.loss_type.startswith("BPR"):
            self.loss_top_k = int(self.loss_type[len("BPR") :])
            self.loss_type = "BPR"
            print(
                ">> Fine-tuning with BPR loss over top {} predictions".format(
                    self.loss_top_k
                )
            )
        else:
            raise NotImplementedError(
                'Fine-tuning loss "{}" not supported'.format(self.loss_type)
            )

        # Training parameters - aspect loss
        self.aspect_loss_weight = getattr(self.config, "aspect_loss", 0.0)
        if self.aspect_loss_weight > 0.0:
            print(
                ">> Fine-tuning with Aspect BCE loss @ weight {}".format(
                    self.aspect_loss_weight
                )
            )
        else:
            print(">> Fine-tuning with NO aspect loss")

        # Training parameters - discount factor
        self.discount_factor = getattr(self.config, "discount", 1.0)
        if self.discount_factor == 1.0:
            print(">> Not using reward discounting")
        elif self.discount_factor == "inv":
            print(">> Discounting reward by 1/turn")
        else:
            print(">> Discounting reward by {} each turn".format(self.discount_factor))
        self.max_turns = self.config.max_turns
        print(">> Simulating up to {:,} turns per dialog".format(self.max_turns))

        # Critiquing behavior
        self.behavior = getattr(self.config, "behavior", "coop")
        self.fb_type = getattr(self.config, "fb_type", "N")
        print(
            '>> Bot-play with "{}" user and {} feedback'.format(
                self.behavior, self.fb_type
            )
        )

        # Load numpy model file
        self.pt_ckpt = os.path.join(self.config.pretrained, "model.pt")
        self.model = PLRecWithAspectsKPTorch.from_pretrained_numpy(self.pt_ckpt)
        self.model_kwargs = self.model.train_kwargs

        print(
            ">> Initialized {} from {} with args:\n{}".format(
                self.model.__class__.__name__,
                self.pt_ckpt,
                json.dumps(self.model_kwargs, indent=2, default=str),
            )
        )

        # Freezing weights if necessary
        orig_weight_count = count_parameters(self.model)
        if self.config.freeze.lower() in {"freeze", "true"}:
            for module in [self.model.gamma_i, self.model.gamma_u, self.model.kp_proj]:
                for p in module.parameters():
                    p.requires_grad = False
            print(
                ">> Freezing weights, only fine-tuning ({:,} ->) {:,} aspect encoder weights".format(
                    orig_weight_count, count_parameters(self.model)
                )
            )
        else:
            print(">> Fine tuning all {:,} weights".format(orig_weight_count))

        print(
            "Created {} ({}) with {:,} params".format(
                self.__class__.__name__,
                self.model.__class__.__name__,
                count_parameters(self),
            )
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def critique_aspect(self, critique_vector, u_ix, kp_ix, fb_type):
        val = critique_vector[u_ix, kp_ix] or 1
        if fb_type == "N":
            critique_vector[u_ix, kp_ix] = critique_vector[u_ix, kp_ix] - val
        else:
            critique_vector[u_ix, kp_ix] = critique_vector[u_ix, kp_ix] + val
        return critique_vector

    def training_step(self, batch, batch_idx, is_eval: bool = False):
        # u: B
        # gold: B
        # target_aspects: B x Na <- ground truth for generation
        # u_kps: B x Na
        u, gold, _, _, target_aspects, _, user_kps = batch
        batch_size = u.shape[0]

        # https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning
        # 0.9 approximates max session 10 exp(-1/10)
        turn_multiplier = 1.0
        loss = 0
        total_rec_loss = 0
        total_kp_loss = 0
        critiqued_aspects = [set() for _ in range(batch_size)]

        # Tracking - turns taken to reach gold
        turns_taken = [-1 for _ in range(batch_size)]
        # Gold ranks per turn - T x B
        gold_ranks = []
        # Invalid items (training items)
        invalid_items = (
            [
                [
                    item
                    for item in self.train_ui[u_ix.item()]
                    if item != gold[b_ix].item()
                ]
                for b_ix, u_ix in enumerate(u)
            ]
            if is_eval
            else None
        )

        # Simulate a conversation with critiques from user
        # Repeat critique vector from B x Na to T x B x na
        n_critiques_made = 0.0
        critique_vector = user_kps.clone().detach()
        for turn_ix in range(self.max_turns):
            # Set discount factor
            if self.discount_factor == "inv":
                turn_multiplier = 1 / (1 + turn_ix)
            elif self.discount_factor and turn_ix > 0:
                turn_multiplier *= self.discount_factor
            else:
                turn_multiplier = 1.0

            # Each turn - get scores B x Ni
            turn_scores, _, latent_u = self.model.get_scores(
                u=u, user_kps=critique_vector, invalid=invalid_items, return_latent=True
            )
            check_nan(turn_scores, "turn_scores")
            check_nan(latent_u, "latent_u")

            # Get recommended item & its keyphrases
            rec_items = torch.argmax(turn_scores, dim=-1)
            _, kp_logits, kp_bin, _, _ = self.model.get_kps(
                u=u, i=rec_items, user_kps=critique_vector, latent_u=latent_u
            )
            check_nan(kp_logits, "kp_logits")

            # Track if gold is found, tensor size B
            found_gold = gold == rec_items
            for b_ix, f in enumerate(found_gold):
                if f and turns_taken[b_ix] == -1:
                    turns_taken[b_ix] = turn_ix + 1

            # Gold ranks (for validation tracking) - size B,
            if is_eval:
                turn_ranks_by_item = torch.argsort(
                    torch.argsort(turn_scores, dim=-1, descending=True),
                    dim=-1,
                    descending=False,
                )
                t_gold_ranks = turn_ranks_by_item[
                    torch.arange(batch_size).to(gold.device), gold
                ]
                gold_ranks.append(t_gold_ranks.detach().cpu().numpy().tolist())

            # No need to build critique at the last turn
            if turn_ix == self.max_turns - 1:
                break

            # Build critique
            critique_vector = critique_vector.clone().detach()
            neg_kp_candidates = (kp_bin - target_aspects) == 1
            pos_kp_candidates = (target_aspects - kp_bin) == 1
            for u_ix in range(batch_size):
                # Ignore if gold is found (no more critiques given)
                if found_gold[u_ix]:
                    continue

                # Initialize candidates
                already_critiqued = list(critiqued_aspects[u_ix])
                if already_critiqued:
                    neg_kp_candidates[u_ix, already_critiqued] = False
                    pos_kp_candidates[u_ix, already_critiqued] = False
                neg_cand = torch.where(neg_kp_candidates[u_ix])[0]
                pos_cand = torch.where(pos_kp_candidates[u_ix])[0]
                n_neg = len(neg_cand)
                n_pos = len(pos_cand)

                if self.fb_type == "N" and n_neg <= 0:
                    continue
                if self.fb_type == "P" and n_pos <= 0:
                    continue
                if self.fb_type in {"NP", "PN"} and n_neg <= 0 and n_pos <= 0:
                    continue

                # Random
                if not is_eval and self.behavior == "random":
                    kp_ix, turn_crit_type = None, None
                    if self.fb_type == "N" and n_neg > 0:
                        cand_indices = list(range(n_neg))
                        kp_ix = neg_cand[random.choice(cand_indices)].item()
                        turn_crit_type = "N"
                    elif self.fb_type == "P" and n_pos > 0:
                        cand_indices = list(range(n_pos))
                        kp_ix = pos_cand[random.choice(cand_indices)].item()
                        turn_crit_type = "P"
                    elif self.fb_type == "NP" and (n_neg + n_pos) > 0:
                        cand_indices = list(range(n_neg + n_pos))
                        cand_ix = random.choice(cand_indices)
                        if cand_ix >= n_neg:
                            kp_ix = pos_cand[cand_ix - n_neg].item()
                            turn_crit_type = "P"
                        else:
                            kp_ix = neg_cand[cand_ix].item()
                            turn_crit_type = "N"
                    if kp_ix is not None:
                        critique_vector = self.critique_aspect(
                            critique_vector, u_ix, kp_ix, turn_crit_type
                        )
                        critiqued_aspects[u_ix].add(kp_ix)
                        n_critiques_made += 1

                # Cooperative (validate with cooperative)
                elif is_eval or self.behavior == "coop":
                    if self.fb_type == "N":
                        kp_order = self.kp_pop_order
                    elif self.fb_type == "P":
                        kp_order = self.kp_pop_order[::-1]
                    elif self.fb_type in {"NP", "PN"}:
                        neg_kp_order = self.kp_pop_order
                        pos_kp_order = self.kp_pop_order[::-1]

                    critiqued = False
                    for pop_ix in range(len(self.kp_pop_order)):
                        if self.fb_type in {"NP", "PN"}:
                            kp_ix_neg = neg_kp_order[pop_ix]
                            kp_ix_pos = pos_kp_order[pop_ix]
                        else:
                            kp_ix = kp_order[pop_ix]

                        # If it is a valid negative critique, use it
                        if self.fb_type == "N" and neg_kp_candidates[u_ix, kp_ix]:
                            critique_vector = self.critique_aspect(
                                critique_vector, u_ix, kp_ix, fb_type="N"
                            )
                            # Track
                            critiqued_aspects[u_ix].add(kp_ix)
                            n_critiques_made += 1
                        elif self.fb_type == "P" and pos_kp_candidates[u_ix, kp_ix]:
                            critique_vector = self.critique_aspect(
                                critique_vector, u_ix, kp_ix, fb_type="P"
                            )
                            # Track
                            critiqued_aspects[u_ix].add(kp_ix)
                            n_critiques_made += 1
                            # print(
                            #     "Turn {} | user {} - critiqued aspect {}".format(
                            #         turn_ix, u_ix, kp_ix
                            #     )
                            # )
                            critiqued = True
                        elif self.fb_type in {"NP", "PN"}:
                            if neg_kp_candidates[u_ix, kp_ix_neg]:
                                critique_vector = self.critique_aspect(
                                    critique_vector, u_ix, kp_ix_neg, fb_type="N"
                                )
                                # Track
                                critiqued_aspects[u_ix].add(kp_ix_neg)
                                n_critiques_made += 1
                                critiqued = True
                            elif pos_kp_candidates[u_ix, kp_ix_pos]:
                                critique_vector = self.critique_aspect(
                                    critique_vector, u_ix, kp_ix_pos, fb_type="P"
                                )
                                # Track
                                critiqued_aspects[u_ix].add(kp_ix_pos)
                                n_critiques_made += 1
                                critiqued = True

                        if critiqued:
                            break

            # Calculate loss for the turn & discount if necessary
            if self.loss_type == "CE":
                turn_loss_rec = (
                    F.cross_entropy(input=turn_scores, target=gold) * turn_multiplier
                )
            elif self.loss_type == "BPR":
                # Get top scores
                # B x k, first element is highest score
                top_k_recs = torch.argsort(turn_scores, dim=-1, descending=True)[
                    :, : self.loss_top_k
                ]
                # If gold item is in the top K, ignore it when computing ranking loss
                rank_loss_mask = (
                    ~(top_k_recs == gold.unsqueeze(1))
                ).float()  # True if should be counted
                # BPR loss is log(sigmoid(x_ui - x_uj))
                gold_scores = turn_scores[
                    torch.arange(gold.shape[0]).to(gold.device), gold
                ]
                # Compute batchwise loss
                top_k_scores = torch.stack(
                    [
                        turn_scores[b_ix, top_recs]
                        for b_ix, top_recs in enumerate(top_k_recs)
                    ]
                )
                turn_loss_rec = (
                    -F.logsigmoid(-(top_k_scores - gold_scores.unsqueeze(-1)))
                    * rank_loss_mask.float()
                ).mean() * turn_multiplier
            check_nan(turn_loss_rec, "turn_loss_rec")

            # Turn loss - aspect-based
            if self.aspect_loss_weight > 0.0:
                try:
                    turn_loss_rationale = (
                        F.binary_cross_entropy_with_logits(
                            input=kp_logits, target=target_aspects.float()
                        )
                        * turn_multiplier
                    )
                except:
                    print("[[ERROR COMPUTING turn_loss_rationale]]")
                    print("kp_logits: {}, {}".format(type(kp_logits), kp_logits.shape))
                    print(
                        "target_aspects: {}, {}".format(
                            type(target_aspects), target_aspects.shape
                        )
                    )
            else:
                turn_loss_rationale = 0.0
            check_nan(turn_loss_rationale, "turn_loss_rationale")

            turn_loss = turn_loss_rec + turn_loss_rationale
            check_nan(turn_loss, "turn_loss")
            total_rec_loss += turn_loss_rec
            total_kp_loss += turn_loss_rationale
            loss += turn_loss

        # Verify loss values
        check_nan(loss, "loss")

        # Fill in turns taken
        turns_taken = [t if t != -1 else self.max_turns for t in turns_taken]

        # print(
        #     "{} critiques made total for {} users ({}/user) over {:,} max turns".format(
        #         n_critiques_made,
        #         batch_size,
        #         n_critiques_made / batch_size,
        #         self.max_turns,
        #     )
        # )

        log = {
            "loss": loss.detach(),
            "rec_loss": total_rec_loss.detach(),
            "kp_loss": total_kp_loss.detach(),
            "crit_per_user": n_critiques_made / batch_size,
        }
        for k, v in log.items():
            self.log(k, v, prog_bar=True, logger=True, batch_size=batch_size)

        return_dict = {
            "loss": loss,
            # Everything in this dictionary logs
            "log": log,
            # Everything in this dictionary is displayed on the progress bar
            "progress_bar": {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in log.items()
            },
        }
        # return_dict.update(log)

        # Store additional values for logging
        if is_eval:
            # Tracking turns
            return_dict["turns"] = turns_taken
            return_dict["ranks"] = gold_ranks

        return return_dict

    def eval_step(self, batch, batch_idx: int, split: str):
        # No difference in train vs validation
        metrics = self.training_step(batch, batch_idx, is_eval=True)

        metrics_dict = {
            "val_" + k: v
            for k, v in metrics.items()
            if k not in {"log", "progress_bar"}
        }

        # Log dict
        for log in ["log", "progress_bar"]:
            metrics_dict[log] = {
                "val_" + k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in metrics[log].items()
            }

        return metrics_dict

    def eval_epoch_end(self, outputs, split: str):
        """
        To report:
            Validation loss
            MRR@1/5/10 - MRR @ 1/5/10 turns
            Hr@1/5/10 - % of conversations reaching target within 1/5/10 turns
            SR@1/5/10 - % of conversations reaching gold rank 1/5/10 within max turns
            TT1/5/10 - Turns to reach GR1/5/10
        """
        # Accumulated metrics
        total_loss = 0
        turns_taken = []
        gold_ranks = []

        for o in outputs:
            total_loss += o[f"{split}_loss"]
            turns_taken.extend(o[f"{split}_turns"])
            # Gold ranks for each output is a T x B list
            # and we need to reshape to B x T to extend
            out_gold_ranks = o[f"{split}_ranks"]
            # Split turns is a B x 1 array, so of the correct shape for us
            for b in range(len(o[f"{split}_turns"])):
                obs_ranks = [out_gold_ranks[t][b] for t in range(self.max_turns)]
                gold_ranks.append(obs_ranks)

        # LOGGING
        metrics = dict()
        log_step = f"{self.global_step},{self.current_epoch}"
        print(f"\n==== [Epoch {self.current_epoch} ({self.global_step:,} steps]")

        # LOSS
        avg_loss = total_loss / len(outputs)
        log_step += f",{float(avg_loss)}"
        print(f"Loss: {float(avg_loss)}")
        metrics[f"{split}_loss"] = avg_loss

        # MRR / mean reciprocal rank at T turns
        for at_t in [1, 5, 10]:
            if at_t >= self.max_turns:
                continue
            mrr_at_t = np.mean([1 / (g[at_t - 1] + 1) for g in gold_ranks])
            log_step += f",{float(mrr_at_t)}"
            print(f"MRR@{at_t}: {float(mrr_at_t)}")
            metrics[f"mrr@{at_t}"] = torch.FloatTensor([mrr_at_t]).squeeze()

        # HR / hit rate: % of conversations reaching target within T turns
        for at_t in [1, 5, 10]:
            if at_t >= self.max_turns:
                continue
            hr_at_t = np.mean([float(min(g[:at_t]) < 1) for g in gold_ranks])
            log_step += f",{float(hr_at_t)}"
            print(f"HR@{at_t}: {float(hr_at_t) * 100.0:.2f}%")
            metrics[f"hr@{at_t}"] = torch.FloatTensor([hr_at_t]).squeeze()

        # SR / success rate: % of conversations reaching gold rank < R
        for at_r in [1, 5, 10]:
            if at_t >= self.max_turns:
                continue
            sr_at_r = np.mean([float(min(g) < at_r) for g in gold_ranks])
            log_step += f",{float(sr_at_r)}"
            print(f"SR@{at_r}: {float(sr_at_r) * 100.0:.2f}%")
            metrics[f"sr@{at_r}"] = torch.FloatTensor([sr_at_r]).squeeze()

        # TT / turns to : Avg. turns to reach gold rank < R
        for at_r in [1, 5, 10]:
            if at_t >= self.max_turns:
                continue
            turns_to_r = []
            for g in gold_ranks:
                ttr = self.max_turns
                for turn_ix, rank in enumerate(g):
                    if rank < at_r:
                        ttr = turn_ix + 1
                        break
                turns_to_r.append(ttr)
            turns_to_r = np.mean(turns_to_r)
            log_step += f",{float(turns_to_r)}"
            print(f"Turns to {at_r}: {float(turns_to_r)}")
            metrics[f"tt{at_r}"] = torch.FloatTensor([turns_to_r]).squeeze()

        print("--------------------------\n")

        # REPORT METRICS
        with open(self.metrics_loc, "a+") as wf:
            _ = wf.write(log_step)
            _ = wf.write("\n")

        log_dict_detached = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        return {**metrics, "log": log_dict_detached, "progress_bar": log_dict_detached}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.config.fixed_lr:
            scaled_lr = self.config.lr
            print("FIXED learning rate at {}".format(scaled_lr))
        else:
            effect_bsz = self.num_gpus * self.config.batch_size * self.config.grad_acc
            scaled_lr = self.config.lr * effect_bsz
            print("SCALED learning rate to {}".format(scaled_lr))

        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer == "lamb":
            optimizer = Lamb(
                optim_params, lr=scaled_lr, weight_decay=self.config.l2_lambda
            )
        elif self.config.optimizer == "adam":
            optimizer = Adam(
                optim_params, lr=scaled_lr, weight_decay=self.config.l2_lambda
            )
        elif self.config.optimizer == "radam":
            optimizer = RAdam(
                optim_params, lr=scaled_lr, weight_decay=self.config.l2_lambda
            )
        else:
            raise NotImplementedError(
                f'Optimizer "{self.config.optimizer}" not supoprted!'
            )
        return optimizer
