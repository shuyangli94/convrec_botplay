import json
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from convrec.models import BPRItemModel
from convrec.optimizer import Adam, Lamb, RAdam
from convrec.utils import check_nan, count_parameters, debug_log


class System(pl.LightningModule):
    saved_metrics = [
        'global_step', 'epoch', 'loss', 'med_rank', 'kp_p', 'kp_r', 'kp_f1',
        'auc', 'mrr', 'hr@1', 'hr@20', 'hr@50', 'NDCG@10', 'NDCG@20', 'NDCG@50'
    ]
    choice_metric = 'auc'

    def __init__(self,
                 config,
                 train_ui: dict = None,
                 all_item_kp: list = None,
                 **kwargs):
        super().__init__()

        # Store configurations
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.debug_mode = self.config.debug
        if self.debug_mode:
            print('~~~ DEBUG MODE ~~~')

        # Metrics
        self.metrics_loc = os.path.join(self.config.ckpt_path,
                                        self.config.name, 'metrics.csv')
        print('\n>> Saving metrics to {}'.format(self.metrics_loc))

        # Training UI if being used to compute metrics
        self.train_ui = train_ui
        if self.train_ui:
            print(
                'Evaluating metrics (HR, MRR, etc.) by removing training items'
            )
        self.all_item_kp = all_item_kp
        if self.all_item_kp is not None:
            self.all_item_kp = torch.FloatTensor(self.all_item_kp)
            print(
                'Evaluating ranking metrics using keyphrases extracted from all items ({})'.
                format(self.all_item_kp.shape))

        # BPR model
        self.model_kwargs = {
            'k': self.config.model_type.k,
            'n_users': self.config.n_users,
            'n_items': self.config.n_items,
            'n_kp': self.config.n_kp,
            'use_item_emb': self.config.model_type.item_emb,
        }
        if self.config.kp_weight == 0:
            print(
                '\n\n----\nRUNNING WITH n_kp = 0 -> NO ASPECT PREDICATION!\n----\n\n'
            )
        print('Creating BPRItemModel model with kwargs:\n{}'.format(
            json.dumps(self.model_kwargs, indent=2, default=str)))
        self.model = BPRItemModel(**self.model_kwargs)

        print('Created {} ({}) with {:,} params'.format(
            self.__class__.__name__, self.model.__class__.__name__,
            count_parameters(self)))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, is_eval: bool = False):
        # U, I, J -> [B,]
        # aspects -> [B, Na]
        # I/J aspects -> [B, Na]
        u, i, j, aspects, i_kp, j_kp = batch

        try:
            # x_ui -> [B,]
            # x_uj -> [B,]
            # aspect_ui -> [B, Na]
            # losses -> []
            x_ui, x_uj, aspect_ui, bpr_loss, aspect_loss = self.model(
                u, i, j, i_kp, j_kp, aspects)
            check_nan(bpr_loss, 'bpr_loss')
            check_nan(aspect_loss, 'aspect_loss')

            # Compute loss
            if self.config.n_kp == 0:
                loss = bpr_loss
            else:
                loss = bpr_loss + self.config.kp_weight * aspect_loss
        except:
            print('Inputs:')
            debug_log(u, 'u', debug=True)
            debug_log(i_kp, 'i_kp', debug=True)
            debug_log(j_kp, 'j_kp', debug=True)
            debug_log(aspects, 'aspects', debug=True)
            raise

        log = {
            'loss': loss,
        }
        return_dict = {
            'loss': loss,
            # Everything in this dictionary logs
            'log': log,
            # Everything in this dictionary is displayed on the progress bar
            'progress_bar': {k: v
                             for k, v in log.items() if 'loss' not in k},
        }

        # Store additional values for stuff
        if is_eval:
            # AUC
            return_dict['u'] = u.detach().cpu()
            return_dict['corr'] = (x_ui > x_uj).detach().cpu()

            # Aspect P/R/F1
            if self.config.kp_weight != 0:
                pred_kp_bin = torch.bernoulli(
                    torch.sigmoid(aspect_ui)).detach()
                n_pos_true = (aspects == 1).sum(dim=-1).detach().cpu().numpy()
                n_pos_pred = (
                    pred_kp_bin == 1).sum(dim=-1).detach().cpu().numpy()
                n_both_true = (
                    (aspects == 1) &
                    (pred_kp_bin == 1)).sum(dim=-1).detach().cpu().numpy()
                # Scores -> [B, ]
                precision = np.nan_to_num(n_both_true / n_pos_pred, 0.0)
                recall = np.nan_to_num(n_both_true / n_pos_true, 1.0)
                f1 = np.nan_to_num(
                    2 * precision * recall / (precision + recall), 0.0)
                return_dict.update({
                    'kp_p': precision,
                    'kp_r': recall,
                    'kp_f1': f1
                })

            # Compute rankings
            invalid = None
            if self.train_ui:
                invalid = [
                    torch.LongTensor(list(self.train_ui[uu]))
                    for uu in u.detach().cpu().numpy()
                ]
            # Move to correct device
            if self.all_item_kp.device != u.device:
                self.all_item_kp = self.all_item_kp.to(u.device)
            # Get scores & # candidates
            item_scores, n_candidates = self.model.get_scores(
                u, invalid=invalid, all_item_kp=self.all_item_kp)
            # item_ranks[b, i] = rank of item i
            item_ranks = self.model.get_ranks(
                u,
                item_scores=item_scores,
                all_item_kp=self.all_item_kp,
                index_by_item=True)
            gold_ranks = item_ranks[torch.arange(item_ranks.size(0)), i]
            return_dict['ranks'] = gold_ranks.detach().cpu()
            return_dict['n_cand'] = n_candidates

        return return_dict

    def eval_step(self, batch, batch_idx: int, split: str):
        # No difference in train vs validation
        metrics = self.training_step(batch, batch_idx, is_eval=True)

        metrics_dict = {
            'val_' + k: v
            for k, v in metrics.items() if k not in {'log', 'progress_bar'}
        }

        # Log dict
        for log in ['log', 'progress_bar']:
            metrics_dict[log] = {
                'val_' + k: v
                for k, v in metrics[log].items()
            }

        return metrics_dict

    def eval_epoch_end(self, outputs, split: str):
        # Accumulate
        all_u = []
        all_corr = []
        all_ranks = []
        all_n_cand = []
        all_kp_p = []
        all_kp_r = []
        all_kp_f1 = []

        total_loss = 0
        for o in outputs:
            all_u.append(o[f'{split}_u'])
            all_corr.append(o[f'{split}_corr'])
            all_ranks.append(o[f'{split}_ranks'])
            all_n_cand.extend(o[f'{split}_n_cand'])
            total_loss += o[f'{split}_loss']

            if self.config.kp_weight != 0:
                all_kp_p.append(o[f'{split}_kp_p'])
                all_kp_r.append(o[f'{split}_kp_r'])
                all_kp_f1.append(o[f'{split}_kp_f1'])

        # Put them into DF-able form
        all_u = torch.cat(all_u).numpy()
        all_corr = torch.cat(all_corr).numpy()
        all_ranks = torch.cat(all_ranks)
        if self.config.kp_weight != 0:
            all_kp_p = np.concatenate(all_kp_p)
            all_kp_r = np.concatenate(all_kp_r)
            all_kp_f1 = np.concatenate(all_kp_f1)

        # LOGGING
        metrics = dict()
        log_step = f'{self.global_step},{self.current_epoch}'
        print(
            f'\n==== [Epoch {self.current_epoch} ({self.global_step:,} steps]')

        # LOSS
        avg_loss = total_loss / len(outputs)
        log_step += f',{float(avg_loss)}'
        print(f'Loss: {float(avg_loss)}')
        metrics[f'{split}_loss'] = float(avg_loss)

        # Mean/median rank
        mean_rank = all_ranks.float().mean()
        med_rank = all_ranks.float().median()
        log_step += f',{float(med_rank)}'
        print('Mean gold rank: {:,.2f}, Median gold rank: {:,.2f}'.format(
            float(mean_rank), float(med_rank)))
        metrics[f'{split}_rank'] = med_rank

        # Aspect P/R/F1, macro-averaged
        if self.config.kp_weight != 0:
            log_step += f',{all_kp_p.mean()},{all_kp_r.mean()},{all_kp_f1.mean()}'
            print(
                f'Aspect P: {all_kp_p.mean()}, R: {all_kp_r.mean()}, F1: {all_kp_f1.mean()}'
            )
        else:
            log_step += f',NA,NA,NA'
            print('No aspect prediction...')

        # AUC
        # Pandas magic to do mean-of-means
        df = pd.DataFrame({
            'u': all_u,
            'gold': all_ranks.numpy(),
            'n_cand': all_n_cand
        })
        df['corr'] = (df['n_cand'] - df['gold']) / df['n_cand']
        try:
            auc = df.groupby(['u'])['corr'].mean().mean()
        except:
            print('Problem DF:')
            print(df)
            raise
        log_step += f',{auc}'
        print(f'AUC: {auc}')
        # Convert to tensor for model checkpoint saving
        metrics['auc'] = torch.FloatTensor([auc]).squeeze()

        # MRR
        mrr = (1 / (all_ranks + 1).float()).mean()
        log_step += f',{float(mrr)}'
        print(f'MRR: {float(mrr)}')
        # metrics['MRR'] = mrr

        # HR@K
        for k in [1, 20, 50]:
            hr_k = (all_ranks < k).float().mean()
            log_step += f',{float(hr_k)}'
            print(f'HR@{k}: {float(hr_k)}')
            if k == 50:
                metrics[f'HR@{k}'] = hr_k

        # NDCG@K - http://people.tamu.edu/~jwang713/pubs/HyperRec-sigir2020.pdf
        # NDCG@K = 1/(log_2(1+rank_i)) if rank_i < K else 0
        for k in [10, 20, 50]:
            ndcg_k = (1 / torch.log2(all_ranks.float() + 2) *
                      (all_ranks < k)).mean()
            log_step += f',{float(ndcg_k)}'
            print(f'NCDG@{k}: {float(ndcg_k)}')
            # if k == 50:
            #     metrics[f'NDCG@{k}'] = ndcg_k

        print('--------------------------\n')

        # REPORT METRICS
        with open(self.metrics_loc, 'a+') as wf:
            _ = wf.write(log_step)
            _ = wf.write('\n')

        return {**metrics, 'log': metrics, 'progress_bar': metrics}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        if self.config.fixed_lr:
            scaled_lr = self.config.lr
            print('FIXED learning rate at {}'.format(scaled_lr))
        else:
            effect_bsz = self.num_gpus * self.config.batch_size * self.config.grad_acc
            scaled_lr = self.config.lr * effect_bsz
            print('SCALED learning rate to {}'.format(scaled_lr))

        if self.config.optimizer == 'lamb':
            optimizer = Lamb(
                self.model.parameters(),
                lr=scaled_lr,
                weight_decay=self.config.l2_lambda)
        elif self.config.optimizer == 'adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=scaled_lr,
                weight_decay=self.config.l2_lambda)
        elif self.config.optimizer == 'radam':
            optimizer = RAdam(
                self.model.parameters(),
                lr=scaled_lr,
                weight_decay=self.config.l2_lambda)
        else:
            raise NotImplementedError(
                f'Optimizer "{self.config.optimizer}" not supoprted!')
        return optimizer
