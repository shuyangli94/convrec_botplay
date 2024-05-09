import datetime as dt
import inspect
import math
import os
import io
import pickle
import pickle5
import stat
import time
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig


def _save(data: object, path: str, msg: str = "", verbose: bool = True):
    start = datetime.now()
    if isinstance(data, pd.DataFrame):
        data.to_pickle(path)
    else:
        with open(path, "wb") as wf:
            pickle.dump(data, wf)
    if verbose:
        print(
            f"[{datetime.now() - start}] - Dumped {msg} to {path} ({(os.path.getsize(path) / 1024 / 1024):.2f} MB)"
        )


def _load(path: str, msg: str = "", verbose: bool = True):
    start = datetime.now()
    try:
        data = pd.read_pickle(path)
    except:
        with open(path, "rb") as rf:
            data = pickle5.load(rf)
    if verbose:
        print(
            f"[{datetime.now() - start}] - Loaded {msg} from {path} ({(os.path.getsize(path) / 1024 / 1024):.2f} MB)"
        )
    return data


def load_config(path: str) -> DictConfig:
    conf = OmegaConf.load(path)
    return conf


def save_config(config: DictConfig, path: str, overwrite: bool = False):
    if os.path.exists(path) and not overwrite:
        print(
            "Config exists at {} ({:.2f} KB)".format(path, os.path.getsize(path) / 1024)
        )
    else:
        OmegaConf.save(config=config, f=path)
        print(
            "Saved configuration to {} ({:.2f} KB)".format(
                path, os.path.getsize(path) / 1024
            )
        )


def memory_usage_gb():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_used = py.memory_info()[0] / 1024 / 1024 / 1024
    print("Currently using {:.2f} GB RAM".format(memory_used))
    return memory_used


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inv_sqrt_schedule(warmup=1e4):
    # From T5 https://arxiv.org/pdf/1910.10683.pdf
    def f(e):
        return math.sqrt(warmup) / math.sqrt(max(e, warmup))

    return f


def load_checkpoint(checkpoint_path):
    model_weights = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    corrected_model_weights = {}
    for k, v in model_weights.items():
        corrected_model_weights[k.replace("model.", "")] = v
    return corrected_model_weights


def get_last_mod_ckpt(directory, allow_last: bool = True):
    """
    gets a list of files sorted by modified time
    keyword args:
    num_files -- the n number of files you want to print
    directory -- the starting root directory of the search
    """
    modified = []

    for root, sub_folders, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".ckpt"):
                continue
            if not allow_last and fname.endswith("last.ckpt"):
                continue

            unix_modified_time = os.stat(os.path.join(root, fname))[stat.ST_MTIME]
            human_modified_time = dt.datetime.fromtimestamp(
                unix_modified_time
            ).strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.join(root, fname)
            modified.append((human_modified_time, filename))

    modified.sort(key=lambda a: a[0], reverse=True)

    if len(modified) > 0:
        print("Last mod file", modified[0])
        return modified[0][1]

    return None


def debug_log(
    x: object,
    msg: str = "",
    caller_fxn: str = None,
    log_fxn: Callable = print,
    debug: bool = False,
):
    """
    Debug logging of an object. Usage:

    from functools import partial
    debug_fxn = partial(debug_log, debug=True)

    Arguments:
        x (object): Some object to be logged/debugged

    Keyword Arguments:
        caller_fxn (str): Name of calling function. If None, will be inferred.
        msg (str): Custom message to be spawned
        log_fxn (callable): Logging function
    """
    if not debug:
        return

    # Get caller function
    caller_fxn = caller_fxn or inspect.currentframe().f_back.f_code.co_name
    log_str = "\n[{}] {}: {}".format(caller_fxn, msg, type(x))

    # Get shape/length
    if hasattr(x, "__len__"):
        try:
            log_str += " len: {},".format(len(x))
        except:
            pass
    if hasattr(x, "shape"):
        try:
            log_str += " shape: {},".format(x.shape)
        except:
            pass

    # Torch tensor
    if isinstance(x, torch.Tensor):
        # Dtype
        log_str += " dtype {}, ".format(x.dtype)

        # Device
        log_str += ' on device "{}", '.format(x.device)

        # Shape
        t_dim = x.dim()
        log_str += " dim {}, ".format(t_dim)

        if t_dim > 0:
            # Iterables
            n_nan = torch.isnan(x).sum()
            n_zero = (x == 0).sum()
            log_str += " ({:,} NaN, {:,} 0".format(n_nan, n_zero)
            if x.dtype in {torch.float, torch.double, torch.half}:
                n_pinf = (x == np.inf).sum()
                n_ninf = (x == -np.inf).sum()
                log_str += ", {:,} +inf, {:,} -inf".format(n_pinf, n_ninf)
            try:
                log_str += ", {} max value, {} min value".format(x.max(), x.min())
            except:
                pass
        else:
            log_str += " VALUE: {}".format(x.item())

        if x.grad_fn is not None:
            log_str += " Gradient: {}".format(x.grad_fn.__class__.__name__)

        log_str += ")"

    # Numpy array
    elif isinstance(x, np.ndarray):
        # Dtype
        log_str += " dtype {}, shape {}, ".format(x.dtype, x.shape)

        if x.shape != ():
            n_none = (x == None).sum()
            n_zero = (x == 0).sum()
            n_pinf = (x == np.inf).sum()
            n_ninf = (x == -np.inf).sum()
            n_nan = 0

            # Need to loop over nditer
            for xx in np.nditer(x, ["refs_ok"]):
                try:
                    if np.isnan(xx):
                        n_nan += 1
                except:
                    pass

            log_str += " ({:,} NaN, {:,} 0, {:,} +inf, {:,} -inf),".format(
                n_nan, n_zero, n_pinf, n_ninf
            )

    # Arbitrary iterable
    elif isinstance(x, Iterable):
        n_none = 0
        n_nan = 0
        n_pinf = 0
        n_zero = 0
        n_ninf = 0

        for xx in x:
            # None
            if xx is None:
                n_none += 1

            # Zero/Nan/+inf/-inf
            try:
                if xx == 0:
                    n_zero += 1
                if np.isnan(xx):
                    n_nan += 1
                if np.isposinf(xx):
                    n_pinf += 1
                if np.isneginf(xx):
                    n_ninf += 1
            except:
                continue

        log_str += " ({:,} None, {:,} NaN, {:,} 0, {:,} +inf, {:,} -inf),".format(
            n_none, n_nan, n_zero, n_pinf, n_ninf
        )

    # Log it
    log_fxn(log_str)
    return log_str


def check_nan(x, msg):
    if x is None:
        return

    nans_found = False

    # Torch types
    if isinstance(x, torch.Tensor):
        nans = torch.isnan(x).sum()
        if nans > 0:
            nans_found = True
    # Non-torch types
    elif isinstance(x, (float, int, np.float64, np.double, np.int64)):
        if math.isnan(x):
            nans_found = True
    else:
        raise NotImplementedError("Cannot check `nan` for type: {}".format(type(x)))

    if nans_found:
        caller_fxn = inspect.currentframe().f_back.f_code.co_name
        debug_log(x, msg, caller_fxn=caller_fxn, debug=True)
        raise Exception(
            "\n\n\n[{}] - NaNs found in {} ({})!".format(caller_fxn, msg, type(x))
        )


def df_to_csv(df: pd.DataFrame):
    s = io.StringIO()
    df.to_csv(s, index=False)
    print(s.getvalue())


def inhour(elapsed):
    return time.strftime("%H:%M:%S", time.gmtime(elapsed))
