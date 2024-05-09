import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class UIJDataset(Dataset):
    """
    A dataset for ingesting interaction data in U,I,J form
    """

    def __init__(self, file_loc: str, limit: int = None):
        super().__init__()

        # Load data
        start = datetime.now()
        self.data = pd.read_pickle(file_loc)[:limit]
        print('{} - Loaded {:,} interactions from {} ({:.2f} MB)'.format(
            datetime.now() - start, len(self.data), file_loc,
            os.path.getsize(file_loc) / 1024 / 1024))

    def __len__(self):
        return len(self.data)

    def _sample(self, k: int):
        indices = np.random.randint(0, len(self), k)
        return [self[i] for i in indices]

    def __getitem__(self, ix):
        u, i, j = self.data[ix]
        return u, i, j


class UIJCollator():
    def __init__(self):
        pass

    def __call__(self, batch):
        u_s, i_s, j_s = zip(*batch)

        u_tensor = torch.LongTensor(u_s)
        i_tensor = torch.LongTensor(i_s)
        j_tensor = torch.LongTensor(j_s)

        return u_tensor, i_tensor, j_tensor


class UIJDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_loc: str,
                 valid_loc: str,
                 test_loc: str,
                 batch_size: int = 256,
                 workers: int = 0,
                 shuffle_train: bool = True,
                 **kwargs):
        super().__init__()
        self.train_file_loc = train_loc
        self.valid_file_loc = valid_loc
        self.test_file_loc = test_loc
        self.batch_size = batch_size
        self.n_workers = workers
        self.ds_kwargs = kwargs
        self.shuffle_train = shuffle_train

    def setup(self, stage=None):

        if stage == 'fit':
            # TRAINING
            if self.train_file_loc:
                self.train_dset = UIJDataset(self.train_file_loc,
                                             **self.ds_kwargs)
            else:
                print(
                    'No training file specified - `train_dataloader()` will not work!'
                )

            # VALIDATION
            if self.valid_file_loc:
                self.valid_dset = UIJDataset(self.valid_file_loc,
                                             **self.ds_kwargs)
            else:
                print(
                    'No validation file specified - `val_dataloader()` will not work!'
                )
        else:
            if self.test_file_loc:
                self.test_dset = UIJDataset(self.test_file_loc,
                                            **self.ds_kwargs)
            else:
                raise NotImplementedError('No test set exists with labels')

    def get_dataloader(self, dset, shuffle: bool):
        return DataLoader(
            dset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=UIJCollator())

    def train_dataloader(self):
        return self.get_dataloader(self.train_dset, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self.get_dataloader(self.valid_dset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dset, shuffle=False)
