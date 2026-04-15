"""Mean Reciprocal Rank.

Returns 1 / rank of the first relevant chunk (1-indexed), or 0 if none of
the retrieved chunks are relevant. Averaging this across questions gives the
standard MRR.
"""

from ..relevance import is_relevant


def mrr_score(chunks: list, gold_spans: list) -> float:
    for i, c in enumerate(chunks, 1):
        if is_relevant(c, gold_spans):
            return 1.0 / i
    return 0.0
