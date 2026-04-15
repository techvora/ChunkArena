"""Precision@K.

Fraction of the top-K retrieved chunks that are relevant. Returns 0 when K
is 0 to avoid division by zero.
"""

from ..relevance import is_relevant


def precision_at_k(chunks: list, gold_spans: list, k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for c in chunks[:k] if is_relevant(c, gold_spans)) / k
