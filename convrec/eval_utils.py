import numpy as np
from sklearn.metrics import ndcg_score


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


def prf(ref, hyp):
    n_ref = len(ref)
    n_hyp = len(hyp)
    n_match = len(set(ref) & set(hyp))
    p = n_match / n_hyp if n_hyp else 0.0
    r = n_match / n_ref if n_ref else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# Compute NDCG for a single observation
def ndcg_ind(gold_kps: set, logits: np.array):
    # Create reference array
    n_kps = logits.shape[-1]
    true_relevance = np.zeros(n_kps)
    true_relevance[gold_kps] = 1.0

    scores = sigmoid(logits)

    return ndcg_score([true_relevance], [scores])
