import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Tuple, Callable

from convrec.utils import debug_log


def null_log(*args, **kwargs):
    pass


def compute_log_probability(
        logits: torch.FloatTensor,
        targets: torch.LongTensor,
        mask: torch.BoolTensor = None,
        debug_fxn: Callable[[object, str], None] = null_log,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    Compute sum of log probs from model logits

    Arguments:
        logits (torch.FloatTensor): Model output logits (B x T x V)
        targets (torch.LongTensor): Target tokens (B x T)
        mask (torch.BoolTensor): Mask revealing only the utterance tokens (B x T)
        debug_fxn (callable): Logging function

    Returns:
        torch.FloatTensor: Target log probabilities (B x T)
        torch.LongTensor: Number of utterance tokens (1)
    """
    # Get log probability from logits via log softmax
    log_probs = F.log_softmax(logits, dim=-1)
    debug_fxn(log_probs, 'log_probs')
    debug_fxn(targets, 'targets')

    # Extract target token probability - (B x T)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    debug_fxn(target_log_probs, 'target_log_probs')

    # Mask to utterance tokens
    if mask is not None:
        target_log_probs = target_log_probs.masked_select(mask)
        debug_fxn(target_log_probs, 'target_log_probs (masked)')
        n_tokens = mask.sum()
    else:
        n_tokens = target_log_probs.numel()
    debug_fxn(n_tokens, 'n_tokens')

    return target_log_probs, n_tokens


def compute_perplexity(sum_log_probs: torch.FloatTensor, n_tokens: int):
    """
    Computes perplexity from a sum of log probabilities and averages over
    number of tokens.

    Args:
        sum_log_probs (torch.Tensor): Sum of token log probabilities
        n_tokens (torch.LongTensor): Number of tokens

    Returns:
        torch.FloatTensor: Perplexity. Call `.item()` to return scalar.
    """
    return torch.exp(-sum_log_probs / n_tokens)


def compute_ce_loss(logits: torch.FloatTensor,
                    targets: torch.LongTensor,
                    mask: torch.BoolTensor = None,
                    seq_weights: torch.FloatTensor = None,
                    criterion: CrossEntropyLoss = None):
    """
    Compute cross entropy loss from logits
    
    Args:
        logits (torch.FloatTensor): Logits
        targets (torch.LongTensor): Target token IDs
        mask (torch.BoolTensor, optional): Boolean mask - 0 for padding tokens. Defaults to None.
        seq_weights: torch.FloatTensor: weight to scale loss by for tokens in the sequence
        criterion (CrossEntropyLoss, optional): Loss criterion. Defaults to standard criterion with reduction='none'.
    
    Returns:
        torch.FloatTensor: 0-dimensional loss value
    """
    # Default criterion
    if criterion is None:
        criterion = CrossEntropyLoss(reduction='none')

    # Compress and compute
    n_logits = logits.size(-1)
    compressed_logits = logits.view(-1, n_logits)
    compressed_targets = targets.contiguous().view(-1)

    # Target loss, unmasked
    target_loss = criterion(
        compressed_logits,  # Compress to (B * T) x V
        compressed_targets,  # Compress to 1 dimension of size B * T
    )

    # Weighting
    if seq_weights is not None:
        try:
            # Multiply the target loss by the scaling factor for each token
            compressed_seq_weights = seq_weights.contiguous().view(-1)
            target_loss = target_loss * compressed_seq_weights
        except:
            debug_log(target_loss, 'target_loss', debug=True)
            debug_log(seq_weights, 'seq_weights', debug=True)
            raise

    # Masking
    if mask is not None:
        compressed_loss_masks = mask.contiguous().view(-1)
        target_loss = target_loss.masked_select(
            # Only calculate loss for loss-appropriate tokens
            # For validation/test, it will be target tokens only
            # For training, it can be S+Q+T (from scratch), Q+T (finetuning), or T
            compressed_loss_masks)

    # Mean
    target_loss = target_loss.float().mean()
    return target_loss
