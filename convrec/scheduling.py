import nvidia_smi
import time
import os
import torch
import sys
import random
from datetime import datetime
from transformers import BertModel, BertTokenizerFast


def get_gpu_stats(gpu_id: int = None):
    # Get GPU utilization
    if gpu_id is None:
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        if "," in visible:
            visible = visible.split(",")[0].strip()
        gpu_id = int(visible) if visible else 0
    # Query for GPU
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    free_gb = info.free / 1024**3
    used_gb = info.used / 1024**3
    max_gb = info.total / 1024**3
    print(
        "GPU {} free VRAM: {:.2f}/{:.2f} GB ({:.2f}% Free)".format(
            gpu_id, free_gb, max_gb, free_gb / max_gb * 100.0
        )
    )
    nvidia_smi.nvmlShutdown()
    return free_gb, used_gb, max_gb


def queue_gpu(gpu_id: int, free_gb_needed: float, wait_s: int = 300):  # e.g. ['0', '1']
    """
    from scheduling import queue_gpu
    queue_gpu(
        gpu_id=None,
        free_gb_needed=10.5,  # Want 10.5 GB
        wait_s=5,  # Wait for 2 seconds each round
    )
    """
    start = datetime.now()

    # Queue
    while True:
        violated = False

        # Get GPU statistics
        free_gb, used_gb, max_gb = get_gpu_stats(gpu_id)

        # Check if there is enough free memory
        if free_gb < free_gb_needed:
            violated = True

        if not violated:
            break
        print(
            "{} - Waiting for {} more seconds...".format(datetime.now() - start, wait_s)
        )
        time.sleep(wait_s)

    print("Queued {} - Starting job on GPU {}".format(datetime.now() - start, gpu_id))


def claim(s_between_logs: int):
    nvidia_smi.nvmlInit()
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    gpu_id = int(visible) if visible else 0
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    max_gb = info.total / 1024**3
    print("Max GPU memory: {:.2f} GB".format(max_gb))
    nvidia_smi.nvmlShutdown()

    t = BertTokenizerFast.from_pretrained("bert-base-uncased")
    mod = BertModel.from_pretrained("bert-base-uncased")
    mod.cuda()
    max_tok = len(t) - 1

    batch_size = 3
    last_log_time = None

    while True:
        rr = random.random()
        if rr < 0.15:
            time.sleep(0.25)
        batch_size = max(batch_size, 1)
        try:
            rrr = random.random()
            if rrr < 0.35:
                inputs = torch.randint(
                    low=0,
                    high=max_tok,
                    size=(batch_size, random.randint(256, 512)),
                    device=mod.device,
                ).long()
            else:
                inputs = torch.randint(
                    low=0, high=max_tok, size=(batch_size, random.randint(256, 512))
                ).long()
                inputs = inputs.cuda()
            outputs = mod(inputs)
        except Exception as e:
            batch_size = int(batch_size * 0.85)
            do_print = False
            if last_log_time is None:
                last_log_time = datetime.now()
                do_print = True
            elif (datetime.now() - last_log_time).total_seconds() > s_between_logs:
                do_print = True
            if do_print:
                print("{}: Reducing batch size to {}".format(e, batch_size))
                last_log_time = datetime.now()
            else:
                sys.stdout.write(".")
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        if reserved_gb < max_gb * 0.85:
            batch_size += 1
            if last_log_time is None:
                last_log_time = datetime.now()
                do_print = True
            elif (datetime.now() - last_log_time).total_seconds() > s_between_logs:
                do_print = True
            if do_print:
                print(
                    "Using {:.2f}/{:.2f} GB - increasing batch size to {}".format(
                        reserved_gb, max_gb, batch_size
                    )
                )
                last_log_time = datetime.now()
            else:
                sys.stdout.write(".")
