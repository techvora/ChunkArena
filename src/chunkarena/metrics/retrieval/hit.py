"""Hit@K.

Binary indicator per query: 1 if at least one of the top-K retrieved chunks
is relevant, else 0. Averaged across questions to report overall coverage.
"""

from ..relevance import is_relevant


def hit_at_k(chunks: list, gold_spans: list, k: int) -> int:
    return int(any(is_relevant(c, gold_spans) for c in chunks[:k]))
