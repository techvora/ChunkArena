"""Recall@K over gold spans.

For each gold span, checks whether at least one of the top-K retrieved
chunks is relevant to that span alone. The score is the fraction of gold
spans covered in that way. Returns 0 when the question has no gold spans.
"""

from ..relevance import is_relevant


def recall_at_k(chunks: list, gold_spans: list, k: int) -> float:
    if not gold_spans:
        return 0.0
    found = sum(1 for span in gold_spans
                if any(is_relevant(c, [span]) for c in chunks[:k]))
    return found / len(gold_spans)
