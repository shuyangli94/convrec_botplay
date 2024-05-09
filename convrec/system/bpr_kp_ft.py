import copy
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from convrec.optimizer import Adam, Lamb, RAdam
from convrec.utils import (check_nan, count_parameters, debug_log,
                           get_last_mod_ckpt, load_config)


class System(pl.LightningModule):
    saved_metrics = ['global_step', 'epoch', 'loss', 'kp_p', 'kp_r', 'kp_f1']
    choice_metric = 'kp_f1'

    def __init__(self, config, rec_model: nn.Module):
        super().__init__()

        # Store configurations
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.debug_mode = self.config.debug
        if self.debug_mode:
            print('~~~ DEBUG MODE ~~~')

        # LOAD THE RECSYS MODEL
        print('\n\n----------------- LOADING REC MODEL...\n\n')
        self.rec_model = rec_model
        self.model = copy.deepcopy(self.rec_model.kp_proj)
        self.rec_model.eval()
        # Freeze rec model
        for p in self.rec_model.parameters():
            p.requires_grad = False
        # Make sure the aspect model is unfrozen
        for p in self.model.parameters():
            p.requires_grad = True
        print('REC MODEL LOADED + FROZEN ----------\n\n')

        # Metrics
        self.metrics_loc = os.path.join(self.config.ckpt_path,
                                        self.config.name, 'metrics.csv')
        print('\n>> Saving metrics to {}'.format(self.metrics_loc))

        print('Created {} ({}) with {:,} params'.format(
            self.__class__.__name__, self.model.__class__.__name__,
            count_parameters(self)))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, is_eval: bool = False):
        # U, I -> [B,]
        # aspects -> [B, Na]
        u, i, aspects = batch

        try:
            # Latent representations
            with torch.no_grad():
                u_emb = self.rec_model.gamma_u(u)
                i_emb = self.rec_model.gamma_i(i)

            # KP logits: [B, Na]
            kp_logits = self.model(u_emb + i_emb)
            loss = F.binary_cross_entropy_with_logits(
                input=kp_logits, target=aspects.float())
            check_nan(kp_logits, 'kp_logits')
            check_nan(loss, 'loss')
        except:
            print('Inputs:')
            debug_log(u, 'u', debug=True)
            debug_log(i, 'i', debug=True)
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
            # pred_kp_bin = (torch.sigmoid(kp_logits) > 0.5).float().detach()
            pred_kp_bin = torch.bernoulli(torch.sigmoid(kp_logits)).detach()
            n_pos_true = (aspects == 1).sum(dim=-1).detach().cpu().numpy()
            n_pos_pred = (pred_kp_bin == 1).sum(dim=-1).detach().cpu().numpy()
            n_both_true = (
                (aspects == 1) &
                (pred_kp_bin == 1)).sum(dim=-1).detach().cpu().numpy()
            # Scores -> [B, ]
            precision = np.nan_to_num(n_both_true / n_pos_pred, 0.0)
            recall = np.nan_to_num(n_both_true / n_pos_true, 1.0)
            f1 = np.nan_to_num(2 * precision * recall / (precision + recall),
                               0.0)
            return_dict.update({
                'kp_p': precision,
                'kp_r': recall,
                'kp_f1': f1
            })

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
        all_kp_p = []
        all_kp_r = []
        all_kp_f1 = []

        total_loss = 0
        for o in outputs:
            total_loss += o[f'{split}_loss']
            all_kp_p.append(o[f'{split}_kp_p'])
            all_kp_r.append(o[f'{split}_kp_r'])
            all_kp_f1.append(o[f'{split}_kp_f1'])

        # Put them into DF-able form
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

        # Aspect P/R/F1, macro-averaged
        log_step += f',{all_kp_p.mean()},{all_kp_r.mean()},{all_kp_f1.mean()}'
        metrics['kp_p'] = torch.FloatTensor([all_kp_p.mean()]).squeeze()
        metrics['kp_r'] = torch.FloatTensor([all_kp_r.mean()]).squeeze()
        metrics['kp_f1'] = torch.FloatTensor([all_kp_f1.mean()]).squeeze()
        print(
            f'Aspect P: {all_kp_p.mean()}, R: {all_kp_r.mean()}, F1: {all_kp_f1.mean()}'
        )

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
