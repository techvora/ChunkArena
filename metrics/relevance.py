"""Semantic relevance check used by every retrieval metric.

A retrieved chunk is considered relevant to a question if either of the
following holds:

    1. Any gold span appears as a case-insensitive substring of the chunk.
    2. The chunk embedding has cosine similarity at least SOFT_THRESHOLD
       against any gold span embedding.

The substring path is cheap and handles the exact-match case. The semantic
path is the fallback that lets paraphrased gold spans still match. Every
decision is memoized in relevance_cache so the same (chunk, spans) pair is
only evaluated once per run.
"""

from sklearn.metrics.pairwise import cosine_similarity

from config import SOFT_THRESHOLD
from .embedding import get_embedding, relevance_cache


def is_relevant(chunk: str, gold_spans: list) -> bool:
    """Chunk relevant if exact match or cosine similarity >= SOFT_THRESHOLD."""
    key = (chunk, tuple(gold_spans))
    if key in relevance_cache:
        return relevance_cache[key]

    chunk_lower = chunk.lower()
    for span in gold_spans:
        if span.lower() in chunk_lower:
            relevance_cache[key] = True
            return True

    chunk_emb = get_embedding(chunk)
    for span in gold_spans:
        score = cosine_similarity([chunk_emb], [get_embedding(span)])[0][0]
        if score >= SOFT_THRESHOLD:
            relevance_cache[key] = True
            return True

    relevance_cache[key] = False
    return False
