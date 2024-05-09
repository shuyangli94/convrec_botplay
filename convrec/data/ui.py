import json
import random
from datetime import datetime
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from convrec.utils import _load


class UIDataset(Dataset):
    """
    A dataset for ingesting interaction data in U,I form
    """

    def __init__(
            self,
            split_ui: dict,  # (u, i)
            review_kp: dict,  # Keyphrase/aspect indices / (u, i)
            n_kp: int,  # Number of unique keyphrases/aspects
            split: str,
            sample: int = None):
        super().__init__()

        # Parameters
        self.sample = sample
        self.split = split
        self.n_kp = n_kp

        # Load data
        start = datetime.now()
        self.review_kp = review_kp

        # Tuples of (u, i) for the split
        self.index = split_ui
        if self.sample:
            if isinstance(self.sample, float):
                n_sample = int(len(self.index) * self.sample)
            else:
                n_sample = self.sample
            random.shuffle(self.index)
            self.index = self.index[:n_sample]

        print(
            '{} - Created {} {} with {:,} observations and {:,} aspect vocab'.
            format(datetime.now() - start, self.split, self.__class__.__name__,
                   len(self.index), self.n_kp))

    def __len__(self):
        return len(self.index)

    def _sample(self, k: int):
        indices = np.random.randint(0, len(self), k)
        return [self[i] for i in indices]

    def __getitem__(self, ix):
        # Get basic data
        u, i = self.index[ix]

        # Aspects
        aspects = [0 for _ in range(self.n_kp)]
        for a_ix in self.review_kp[(u, i)]:
            aspects[a_ix] = 1

        return u, i, aspects


class UICollator():
    def __init__(self):
        pass

    def __call__(self, batch):
        u_s, i_s, aspects_list = zip(*batch)

        # Tensor-fy
        u = torch.LongTensor(u_s)
        i = torch.LongTensor(i_s)
        aspects = torch.LongTensor(aspects_list)

        return u, i, aspects


class UIDataModule(pl.LightningDataModule):
    def __init__(self,
                 splits_loc: str,
                 kp_loc: str,
                 batch_size: int = 256,
                 workers: int = 0,
                 shuffle_train: bool = True,
                 pin_memory: bool = True,
                 **kwargs):
        super().__init__()
        # File locations
        self.splits_loc = splits_loc
        self.kp_loc = kp_loc

        # Data
        self.review_kp = None
        self.kp_map = None
        self.train_ui = None
        self.valid_ui = None
        self.test_ui = None
        self.user_map = None
        self.item_map = None

        # Params
        self.batch_size = batch_size
        self.n_workers = workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory

        # Assorted kwargs
        self.ds_kwargs = kwargs

        print('Prepared DataModule with kwargs:')
        print(json.dumps(self.ds_kwargs, indent=2, default=str))

    def setup(self, stage=None):
        # Load the data
        self.train_ui, self.valid_ui, self.test_ui = _load(
            self.splits_loc, 'U/I per split')
        self.kp_map, self.review_kp = _load(
            self.kp_loc, 'keyphrase map & keyphrase ix / review')

        # Get User/item maps & convert data
        all_users = sorted(self.train_ui.keys())
        all_items = sorted(set(chain.from_iterable(self.train_ui.values())))
        self.user_map = dict(zip(all_users, list(range(len(all_users)))))
        self.item_map = dict(zip(all_items, list(range(len(all_items)))))
        print('{:,} unique users and {:,} unique items mapped to IX'.format(
            len(self.user_map), len(self.item_map)))
        self.train_ui = {
            self.user_map[u]: {self.item_map[i]
                               for i in i_s}
            for u, i_s in tqdm(
                self.train_ui.items(), total=len(self.train_ui))
        }
        self.valid_ui = {
            self.user_map[u]: self.item_map[i]
            for u, i in tqdm(self.valid_ui.items(), total=len(self.valid_ui))
        }
        self.test_ui = {
            self.user_map[u]: self.item_map[i]
            for u, i in tqdm(self.test_ui.items(), total=len(self.test_ui))
        }
        self.review_kp = {
            (self.user_map[u], self.item_map[i]): kp
            for (u, i), kp in tqdm(
                self.review_kp.items(), total=len(self.review_kp))
            if u in self.user_map and i in self.item_map
        }

        # Create dataset kwargs
        common_kwargs = {
            'review_kp': self.review_kp,
            'n_kp': len(self.kp_map),
        }
        common_kwargs.update(self.ds_kwargs)

        if stage == 'fit':
            # TRAINING
            train_keys = []
            for u, i_s in self.train_ui.items():
                train_keys.extend([(u, i) for i in i_s])
            self.train_dset = UIDataset(
                split_ui=train_keys, split='train', **common_kwargs)

            # VALIDATION
            self.valid_dset = UIDataset(
                split_ui=list(self.valid_ui.items()),
                split='valid',
                **common_kwargs)
        else:
            # TESTING
            self.test_dset = UIDataset(
                split_ui=list(self.test_ui.items()),
                split='test',
                **common_kwargs)

    def get_dataloader(self, dset, shuffle: bool):
        return DataLoader(
            dset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            collate_fn=UICollator())

    def train_dataloader(self):
        return self.get_dataloader(self.train_dset, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self.get_dataloader(self.valid_dset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dset, shuffle=False)
