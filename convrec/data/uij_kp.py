import json
import os
import random
from collections import defaultdict, Counter
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from convrec.utils import _load


def create_aspect_vector(n_kp: int, aspect_ixs: set, use_freq: bool = False):
    aspects = [0 for _ in range(n_kp)]

    # Frequency-based
    if use_freq and isinstance(aspect_ixs, dict):
        for a_ix, count in aspect_ixs.items():
            aspects[a_ix] = count

    # Presence-based
    else:
        for a_ix in aspect_ixs:
            aspects[a_ix] = 1

    return aspects


class UIJKPDataset(Dataset):
    """
    A dataset to build justification examples
    Given a user u, target item i
    Encoder Inputs:
        User history: [n_just] snippets
        Item history: [n_just] snippets from the item
    Decoder target:
        [I would recommend {target item name}.] target review
    """

    def __init__(
            self,
            train_ui: dict,  # Items per user in training
            split_ui: dict,  # (u, i)
            neg_corpus: dict,  # u : valid j's or None
            n_neg_samp: int,  # Sampling range
            review_kp: dict,  # Keyphrase/aspect indices / (u, i)
            item_kp: dict,  # Item ID : Keyphrases (set)
            user_kp: dict,  # User ID : dictionary of keyphrase frequency
            n_kp: int,  # Number of unique keyphrases/aspects
            n_items: int,  # Number of unique items
            split: str,
            sample: int = None,
            **kwargs):
        super().__init__()

        # Params
        self.sample = sample
        self.split = split
        self.n_kp = n_kp
        self.n_items = n_items

        # Load data
        start = datetime.now()
        self.neg_corpus = neg_corpus
        self.n_neg_samp = n_neg_samp
        if self.neg_corpus:
            print('Using subset sampling for j items')
        self.review_kp = review_kp
        self.item_kp = item_kp
        self.user_kp = user_kp
        if self.user_kp:
            print('Retrieving user KPs')
        self.train_ui = {k: set(v) for k, v in train_ui.items()}

        # Tuples of (u, i) for the split
        self.index = split_ui
        if self.sample:
            if isinstance(self.sample, float):
                n_sample = int(len(self.index) * self.sample)
            else:
                n_sample = self.sample
            random.shuffle(self.index)
            self.index = self.index[:n_sample]

        if kwargs:
            print('Ignored extra parameters:\n{}'.format(
                json.dumps(kwargs, indent=2, default=str)))

        print('{} - Created {} {} with {:,} observations and {:,} total items'.
              format(datetime.now() - start, self.split,
                     self.__class__.__name__, len(self.index), self.n_items))

    def __len__(self):
        return len(self.index)

    def _sample(self, k: int):
        indices = np.random.randint(0, len(self), k)
        return [self[i] for i in indices]

    def _get_item_by_key(self, key):
        u, i = key

        # Aspects
        target_aspects = create_aspect_vector(
            self.n_kp, self.review_kp[(u, i)], use_freq=False)

        # Sample negative item
        if self.neg_corpus:
            j_ix = random.randint(0, self.n_neg_samp - 1)
            j = self.neg_corpus[u][j_ix]
        else:
            j = random.randint(0, self.n_items - 1)
            while j in self.train_ui[u]:
                j = random.randint(0, self.n_items - 1)

        # Items represented as aspects
        i_kp = create_aspect_vector(self.n_kp, self.item_kp[i], use_freq=False)
        j_kp = create_aspect_vector(self.n_kp, self.item_kp[j], use_freq=False)

        if self.user_kp:
            u_kp = create_aspect_vector(
                self.n_kp, self.user_kp[u], use_freq=True)
            return u, i, j, target_aspects, i_kp, j_kp, u_kp

        return u, i, j, target_aspects, i_kp, j_kp

    def __getitem__(self, ix):
        # Get basic data
        u, i = self.index[ix]

        return self._get_item_by_key((u, i))


class UIJKPCollator():
    def __init__(self):
        pass

    def __call__(self, batch):
        if len(batch[0]) == 6:
            u_s, i_s, j_s, aspects_list, i_kps, j_kps = zip(*batch)
            u_kps = None
        else:
            u_s, i_s, j_s, aspects_list, i_kps, j_kps, u_kps = zip(*batch)

        # Tensor-fy
        u = torch.LongTensor(u_s)
        i = torch.LongTensor(i_s)
        j = torch.LongTensor(j_s)
        aspects = torch.LongTensor(aspects_list)

        # KPs
        i_kp = torch.LongTensor(i_kps)
        j_kp = torch.LongTensor(j_kps)

        outputs = [u, i, j, aspects, i_kp, j_kp]
        if u_kps is not None:
            outputs.append(torch.FloatTensor(u_kps))

        return outputs


class UIJKPDataModule(pl.LightningDataModule):
    def __init__(
            self,
            splits_loc: str,
            kp_loc: str,
            neg_subset_ratio: float = None,  # Ding et al. 2019
            batch_size: int = 256,
            workers: int = 0,
            shuffle_train: bool = True,
            pin_memory: bool = True,
            min_item_kp: int = None,
            use_user_kp: bool = False,
            **kwargs):
        super().__init__()
        # File locations
        self.splits_loc = splits_loc
        self.kp_loc = kp_loc

        # Data
        self.review_kp = None
        self.item_kp = None
        self.item_kp_train = None
        self.user_kp = None
        self.user_kp_train = None
        self.kp_map = None
        self.train_ui = None
        self.valid_ui = None
        self.test_ui = None
        self.user_map = None
        self.item_map = None
        self.n_neg_sample = None
        self.neg_corpus = None
        self.item_kp_vec = None
        self.stage = None

        # Params
        self.neg_subset_ratio = neg_subset_ratio
        self.batch_size = batch_size
        self.n_workers = workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.min_item_kp = min_item_kp
        self.use_user_kp = use_user_kp

        # Assorted kwargs
        self.ds_kwargs = kwargs

        print('Prepared DataModule with kwargs:')
        print(json.dumps(self.ds_kwargs, indent=2, default=str))

    def limit_kps(self, min_freq: int = None):
        kp_accumulators = [
            self.item_kp, self.item_kp_train, self.user_kp, self.user_kp_train
        ]
        if min_freq:
            kp_accumulators = [{k: v
                                for k, v in acc.items() if v >= min_freq}
                               for acc in kp_accumulators]
        return kp_accumulators

    def setup(self, stage=None):
        # Load the data
        self.user_map, self.item_map, self.train_ui, self.valid_ui, self.test_ui = \
            _load(self.splits_loc, 'maps, U/I per split')
        self.kp_map, self.review_kp = _load(
            self.kp_loc, 'keyphrase map & keyphrase ix / review')
        print('{:,} unique users, {:,} unique items, {:,} aspects'.format(
            len(self.user_map), len(self.item_map), len(self.kp_map)))

        # KPs for each item
        self.item_kp = defaultdict(Counter)
        self.user_kp = defaultdict(Counter)
        print('Creating user/item KPs')
        for (u, i), kp_set in tqdm(
                self.review_kp.items(), total=len(self.review_kp)):
            self.item_kp[i].update(kp_set)
            self.user_kp[u].update(kp_set)
        # Fill in missing aspects if necessary
        self.item_kp = {
            k: dict(self.item_kp[k])
            for k in range(len(self.item_map))
        }
        self.user_kp = {
            k: dict(self.user_kp[k])
            for k in range(len(self.user_map))
        }

        # Training KPs for item/user
        self.item_kp_train = defaultdict(Counter)
        self.user_kp_train = defaultdict(Counter)
        print('Creating training user/item KPs')
        for u, i_s in tqdm(self.train_ui.items(), total=len(self.train_ui)):
            for i in i_s:
                self.item_kp_train[i].update(self.review_kp[(u, i)])
                self.user_kp_train[u].update(self.review_kp[(u, i)])
        # Fill in missing aspects if necessary
        self.item_kp_train = {
            k: dict(self.item_kp_train[k])
            for k in range(len(self.item_map))
        }
        self.user_kp_train = {
            k: dict(self.user_kp_train[k])
            for k in range(len(self.user_map))
        }

        # Limit
        if self.min_item_kp:
            self.item_kp, self.item_kp_train, \
                self.user_kp, self.user_kp_train = self.limit_kps(
                min_freq=self.min_item_kp)

        # Vector of Ni x Na
        self.item_kp_vec = [
            create_aspect_vector(
                n_kp=len(self.kp_map), aspect_ixs=self.item_kp[ix])
            for ix in range(len(self.item_map))
        ]
        n_kps = [len(v) for v in self.item_kp.values()]
        print(
            '{:,} total aspects represented across {:,} items ({:.2f} median/item)'.
            format(sum(n_kps), len(n_kps), np.median(n_kps)))

        # Set up subset negative sampling
        if self.neg_subset_ratio:
            self.neg_corpus = dict()
            self.n_neg_sample = int(len(self.item_map) * self.neg_subset_ratio)
            all_items = set(self.item_map.values())
            print('Setting up negative samples ({:,}/user)'.format(
                self.n_neg_sample))
            for u, i_s in tqdm(
                    self.train_ui.items(), total=len(self.train_ui)):
                invalid = set(i_s) | set(self.valid_ui.get(u, [])) | set(
                    self.test_ui.get(u, []))
                user_valid = list(all_items - invalid)
                assert self.n_neg_sample <= len(user_valid)
                self.neg_corpus[u] = random.sample(user_valid,
                                                   self.n_neg_sample)
            print('Created negative (j) sampling corpus for {:,} users'.format(
                len(self.neg_corpus)))

        # Create dataset kwargs
        common_kwargs = {
            'train_ui': self.train_ui,
            'review_kp': self.review_kp,
            'item_kp': self.item_kp_train,
            'neg_corpus': self.neg_corpus,
            'n_neg_samp': self.n_neg_sample,
            'n_kp': len(self.kp_map),
            'n_items': len(self.item_map),
            'user_kp': self.user_kp_train if self.use_user_kp else None,
        }
        common_kwargs.update(self.ds_kwargs)

        self.stage = stage
        if self.stage == 'fit':
            # TRAINING
            train_keys = []
            for u, i_s in self.train_ui.items():
                train_keys.extend([(u, i) for i in i_s])
            self.train_dset = UIJKPDataset(
                split_ui=train_keys, split='train', **common_kwargs)

            # VALIDATION
            valid_keys = []
            for u, i_s in self.valid_ui.items():
                valid_keys.extend([(u, i) for i in i_s])
            self.valid_dset = UIJKPDataset(
                split_ui=valid_keys, split='valid', **common_kwargs)
        elif self.stage == 'test':
            # TESTING
            test_keys = []
            for u, i_s in self.test_ui.items():
                test_keys.extend([(u, i) for i in i_s])
            self.test_dset = UIJKPDataset(
                split_ui=test_keys, split='test', **common_kwargs)
        else:
            raise NotImplementedError('{} not an acceptable stage'.format(
                self.stage))

    def get_dataloader(self, dset, shuffle: bool):
        return DataLoader(
            dset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            collate_fn=UIJKPCollator())

    @property
    def eval_dataset(self):
        if self.stage == 'fit':
            return self.valid_dset
        elif self.stage == 'test':
            return self.test_dset
        else:
            raise ValueError('Error: invoke "dm.setup(valid/test)" first!')
    
    @property
    def eval_dataloader(self):
        if self.stage == 'fit':
            return self.val_dataloader()
        elif self.stage == 'test':
            return self.test_dataloader()
        else:
            raise ValueError('Error: invoke "dm.setup(valid/test)" first!')

    def train_dataloader(self):
        return self.get_dataloader(self.train_dset, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self.get_dataloader(self.valid_dset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dset, shuffle=False)
