import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from convrec.models import BPR
from convrec.optimizer import Adam, Lamb, RAdam
from convrec.utils import check_nan, count_parameters, debug_log


class System(pl.LightningModule):
    saved_metrics = ['global_step', 'epoch', 'loss', 'auc']

    def __init__(self, config):
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

        # BPR model
        self.model_kwargs = {
            'k': self.config.model_type.k,
            'n_items': self.config.n_items,
            'n_users': self.config.n_users,
        }
        print('Creating BPR model with kwargs:\n{}'.format(
            json.dumps(self.model_kwargs, indent=2, default=str)))
        self.model = BPR(**self.model_kwargs)

        print('Created {} ({}) with {:,} params'.format(
            self.__class__.__name__, self.model.__class__.__name__,
            count_parameters(self)))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, is_eval: bool = False):
        # U, I, J -> [B,]
        u, i, j = batch

        try:
            # x_ui -> [B,]
            # x_uj -> [B,]
            # loss -> []
            x_ui, x_uj, loss = self.model(u, i, j)
            check_nan(loss, 'loss')
        except:
            print('Inputs:')
            debug_log(u, 'u', debug=True)
            debug_log(i, 'i', debug=True)
            debug_log(j, 'j', debug=True)
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

        # Store per-user correct values for AUC calculation
        if is_eval:
            return_dict['u'] = u.detach()
            return_dict['correct'] = (x_ui > x_uj).detach()

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
        total_loss = 0
        for o in outputs:
            all_u.append(o[f'{split}_u'])
            all_corr.append(o[f'{split}_correct'])
            total_loss += o[f'{split}_loss']

        # Put them into DF-able form
        all_u = torch.cat(all_u).cpu().numpy()
        all_corr = torch.cat(all_corr).cpu().numpy()

        # Pandas magic to do mean-of-means
        df = pd.DataFrame({'u': all_u, 'correct': all_corr})
        try:
            auc = df.groupby(['u'])['correct'].mean().mean()
        except:
            print('Problem DF:')
            print(df)
            raise

        avg_loss = total_loss / len(outputs)
        print()
        print('\n[Epoch {}] - Validation AUC: {:.4f}'.format(
            self.current_epoch, auc))
        print('--------------------------')

        # REPORT METRICS
        with open(self.metrics_loc, 'a+') as wf:
            _ = wf.write(
                f'{self.global_step},{self.current_epoch},{float(avg_loss)},{auc}'
            )
            _ = wf.write('\n')

        metrics = {
            f'{split}_loss': total_loss / len(outputs),
            f'{split}_auc': auc,
        }

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
