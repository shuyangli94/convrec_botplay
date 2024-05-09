import json
import os
import shutil
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime
from itertools import product

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tabulate import tabulate

from convrec import system
from convrec.data import UIJKPDataModule
from convrec.utils import df_to_csv, get_last_mod_ckpt, load_config, save_config
from convrec.scheduling import claim


def train_model(config, gpus, multi_gpus: bool, system_class):
    cfg = deepcopy(config)

    # Get data module first
    dm_kwargs = dict(
        splits_loc=cfg.splits_loc,
        kp_loc=cfg.kp_loc,
        batch_size=cfg.batch_size,
        workers=cfg.num_workers,
        neg_subset_ratio=cfg.neg_subset_ratio,
        item_as_kp=False,
        shuffle_train=True,
        use_user_kp=True,
    )
    print("\n!!! CREATING DATAMODULE WITH KWARGS:")
    print(json.dumps(dm_kwargs, indent=2, default=str))
    print("\n")
    dm = UIJKPDataModule(**dm_kwargs)
    dm.setup("fit")

    # Adjust config
    cfg.n_items = len(dm.item_map)
    cfg.n_users = len(dm.user_map)
    cfg.n_kp = len(dm.kp_map)

    # KP popularity
    kp_popularity = Counter()
    for i, kpp in dm.item_kp_train.items():
        kp_popularity.update(kpp)

    KP_POP_ORDER, _ = zip(*kp_popularity.most_common(len(kp_popularity)))

    # Serialize the config
    print("Using config:")
    print(json.dumps(dict(cfg), indent=2, default=str))

    if cfg.load:
        print("Loading model weights...", cfg.load)
        model = system_class.load_from_checkpoint(
            cfg.load,
            config=cfg,
            train_ui=dm.train_ui,
            kp_pop_order=KP_POP_ORDER,
            strict=True,
        )
    else:
        print("!!!!!!    INITIALIZING FROM SCRATCH     !!!!!!!!!!")
        model = system_class(cfg, train_ui=dm.train_ui, kp_pop_order=KP_POP_ORDER)
    choice_metric = model.choice_metric

    # Overwriting old models
    save_path = os.path.join(cfg.ckpt_path, cfg.name)
    if cfg.overwrite and os.path.exists(save_path):
        print(
            '\n\n\nOVERWRITING - MAKE SURE YOU WANT TO DELETE "{}"??'.format(save_path)
        )
        del_true = input("Delete? (Y/N): ")
        if del_true in {"Y", "y"}:
            shutil.rmtree(save_path)
            print('-- Removed "{}" --'.format(save_path))
            os.makedirs(save_path, exist_ok=True)
            print('-- Created empty dir: "{}" --'.format(save_path))
        else:
            print("-- NOT DELETING --")

    resume_ckpt = get_last_mod_ckpt(save_path)
    if resume_ckpt:
        print("Found latest checkpoint file", resume_ckpt)

    if (
        not os.path.exists(model.metrics_loc)
        or os.path.getsize(model.metrics_loc) < 1024
    ):
        progress_callback = pl.callbacks.TQDMProgressBar(
            refresh_rate=1, process_position=0
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=choice_metric,
            mode="max"
            if choice_metric
            in {
                "mrr@1",
                "mrr@5",
                "mrr@10",
                "hr@1",
                "hr@5",
                "hr@10",
                "sr@1",
                "sr@5",
                "sr@10",
            }
            else "min",
            dirpath=save_path,
            save_last=True,
            filename="{epoch:02d}.ckpt",
            save_top_k=1,
        )
        trainer = Trainer(
            logger=None,
            # early_stop_callback=False,
            # log_save_interval=1,
            # checkpoint_callback=ModelCheckpoint(
            #     monitor=choice_metric,
            #     save_top_k=1,
            #     mode="max"
            #     if choice_metric
            #     in {
            #         "mrr@1",
            #         "mrr@5",
            #         "mrr@10",
            #         "hr@1",
            #         "hr@5",
            #         "hr@10",
            #         "sr@1",
            #         "sr@5",
            #         "sr@10",
            #     }
            #     else "min",
            #     save_last=True,
            # ),
            callbacks=[progress_callback, checkpoint_callback],
            # weights_save_path=save_path,
            resume_from_checkpoint=resume_ckpt,
            gpus=gpus,
            # distributed_backend="ddp" if multi_gpus else None,
            precision=16 if cfg.amp else 32,
            max_epochs=cfg.max_epochs,
            max_steps=cfg.max_steps,
            accumulate_grad_batches=cfg.grad_acc,
            val_check_interval=cfg.val_interval,
            replace_sampler_ddp=True,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,
            gradient_clip_val=cfg.grad_clip_val,
        )

        trainer.fit(model, dm)

    # Check metrics
    mdf = pd.read_csv(model.metrics_loc, names=model.saved_metrics).sort_values(
        [choice_metric, "loss"],
        ascending=[
            choice_metric
            not in {
                "mrr@1",
                "mrr@5",
                "mrr@10",
                "hr@1",
                "hr@5",
                "hr@10",
                "sr@1",
                "sr@5",
                "sr@10",
            },
            True,
        ],
    )
    print("\n\n")
    print(tabulate(mdf, headers="keys", tablefmt="psql", showindex=False))
    print()
    df_to_csv(mdf)
    print()
    best_row = mdf.iloc[0].to_dict()
    best_metric = best_row[choice_metric]
    best_loss = best_row["loss"]
    best_epoch = best_row["epoch"]

    return cfg, (choice_metric, best_metric), best_loss, best_epoch, best_row


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    print(cfg)
    cfg = cfg.params
    print("=== RAW CONFIG ===")
    print(json.dumps(dict(cfg), indent=2, default=str))

    # Plug in pretrained config model params
    pt_cfg_loc = cfg.pretrained
    pt_cfg = load_config(os.path.join(pt_cfg_loc, "config.yaml"))
    cfg.model_type = pt_cfg.model_type
    print("=== RAW CONFIG w PRETRAINED MODEL CONFIG ===")
    print(json.dumps(dict(cfg), indent=2, default=str))

    start = datetime.now()

    # Handle CUDA things
    multi_gpus = True
    n_gpus = 0
    vis_devices_os = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis_devices_os:
        gpus = torch.cuda.device_count()
        n_gpus = gpus
        print("\n\n\n===========")
        print("No $CUDA_VISIBLE_DEVICES set, defaulting to {:,}".format(gpus))
        print("===========\n\n\n")
        if gpus < 2:
            multi_gpus = False
        time.sleep(2)
    else:
        gpus = list(map(int, vis_devices_os.split(",")))
        n_gpus = len(gpus)
        if len(gpus) < 2:
            multi_gpus = False
            gpus = 1
        print("Visible devices as specified in $CUDA_VISIBLE_DEVICES: {}".format(gpus))

    SystemClass = getattr(system, cfg.system)

    # GRID SEARCH
    total_grid = 1
    hparams = []
    for hyperparam in [
        "batch_size",
        "discount",
        "freeze",
        "max_turns",
        "lr",
        "l2_lambda",
        "target_item",
        "loss",
        "behavior",
        "fb_type",
        "aspect_loss",
    ]:
        candidates = []
        for c in str(getattr(cfg, hyperparam)).split(","):
            try:
                cc = int(c)
            except:
                try:
                    cc = float(c)
                except:
                    cc = c
            candidates.append((hyperparam, cc))
        hparams.append(candidates)
        print("Searching over {:,} {}".format(len(candidates), hyperparam))
        total_grid *= len(candidates)

    # GRID SEARCH MODEL HYPERPARAMS
    model_hp = []
    for model_hyp, vals in cfg.model_type.items():
        if model_hyp == "model":
            continue
        candidates = []
        for c in str(vals).split(","):
            if c in {"true", "True"}:
                cc = True
            # bool('false') == False apparently
            elif c in {"false", "False"}:
                cc = False
            else:
                try:
                    cc = int(c)
                except:
                    try:
                        cc = float(c)
                    except:
                        cc = c
            candidates.append((model_hyp, cc))
        model_hp.append(candidates)
        print(
            "Searching over {:,} {}: {}".format(len(candidates), model_hyp, candidates)
        )
        total_grid *= len(candidates)

    print("\n\nSearching over a total of {:,} candidate models\n\n".format(total_grid))

    def short_name(full_name):
        if len(full_name) < 4:
            return full_name
        return "".join([s[0] for s in full_name.split("_")]).upper()

    # Tuning loop
    tuning_results = []
    base_name = cfg.name
    for model_hps in product(*model_hp):
        # Configure model to hyperparameters
        model_name = str(base_name)
        for model_hp_name, model_hp_val in model_hps:
            cfg.model_type[model_hp_name] = model_hp_val
            print("Set {} to {}".format(model_hp_name, model_hp_val))
            # Special cases of true/false parameters with default false values
            if (
                model_hp_name in {"item_emb", "no_bias", "target_item", "user_emb"}
                and model_hp_val
            ):
                model_name += f"_{short_name(model_hp_name)}"
            else:
                model_name += f"_{short_name(model_hp_name)}{model_hp_val}"

        for train_hps in product(*hparams):
            # Configure model to training hyperparameters
            cfg.name = model_name
            for train_hp_name, train_hp_val in train_hps:
                cfg[train_hp_name] = train_hp_val
                print("Set {} to {}".format(train_hp_name, train_hp_val))
                if train_hp_name in {"target_item", "freeze"}:
                    cfg.name += f"_{short_name(train_hp_name)}"
                elif train_hp_name == "loss":
                    if train_hp_val != "CE":
                        model_name += f"_{train_hp_val}"
                elif train_hp_name == "aspect_loss":
                    if train_hp_val > 0.0:
                        cfg.name += f"_AL{train_hp_val}"
                elif train_hp_name == "behavior":
                    if train_hp_val != "coop":
                        model_name += f"_{train_hp_val}"
                elif train_hp_name == "fb_type":
                    if train_hp_val != "N":
                        model_name += f"_{train_hp_val}"
                else:
                    cfg.name += f"_{short_name(train_hp_name)}{train_hp_val}"

            # Make sure model directory exists etc., save config
            model_dir = os.path.join(cfg.ckpt_path, cfg.name)
            os.makedirs(model_dir, exist_ok=True)
            cfg_path = os.path.join(model_dir, "config.yaml")
            save_config(cfg, path=cfg_path, overwrite=cfg.overwrite)

            # Train the model
            (
                current_cfg,
                current_metric,
                current_loss,
                current_epoch,
                current_row,
            ) = train_model(
                config=cfg, gpus=gpus, multi_gpus=multi_gpus, system_class=SystemClass
            )

            # Track metrics
            metric_name, metric_val = current_metric
            tuning_dict = deepcopy(current_row)
            tuning_dict["name"] = current_cfg.name
            tuning_dict.update(dict(model_hps))
            tuning_dict.update(dict(train_hps))
            tuning_results.append(tuning_dict)

            # Track how many left
            print(
                "{:,}/{:,} left!".format(total_grid - len(tuning_results), total_grid)
            )

    # Compile stats
    mdf = pd.DataFrame(tuning_results)
    try:
        mdf = mdf.sort_values(
            [metric_name, "loss"],
            ascending=[
                metric_name
                not in {
                    "mrr@1",
                    "mrr@5",
                    "mrr@10",
                    "hr@1",
                    "hr@5",
                    "hr@10",
                    "sr@1",
                    "sr@5",
                    "sr@10",
                },
                True,
            ],
        )
        print("\n\nTOTAL STATS")
        print(tabulate(mdf, headers="keys", tablefmt="psql", showindex=False))
        print()
        df_to_csv(mdf)
        print("{} - DONE".format(datetime.now() - start))
    except:
        print(mdf)
        print(mdf.columns)
        raise

    if getattr(cfg, "claim", False):
        claim(s_between_logs=20)


"""
Bot-Play Fine-tuning for BPR-based models
"""
if __name__ == "__main__":
    main()
