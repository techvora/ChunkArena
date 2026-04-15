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

from chunkarena.config import SOFT_THRESHOLD
from .embedding import get_embedding, relevance_cache


def is_relevant(chunk: str, gold_spans: list) -> bool:
    """Decide whether a chunk is relevant to a question's gold spans.

    A chunk is considered relevant when at least one of the following
    holds: (a) some gold span appears as a case-insensitive substring
    of the chunk, or (b) the chunk embedding has cosine similarity
    ``>= config.SOFT_THRESHOLD`` against at least one gold span
    embedding. The substring path short-circuits the embedding path so
    exact lifts are cheap. Every IR metric in this package
    (hit@k, mrr, precision@k, recall@k, nDCG@k, avg_rank) defers to
    this function, so relevance is a soft cosine match rather than
    exact string equality — the threshold defaults to 0.72 and is
    calibrated for BGE-M3 on the banking corpus. Decisions are
    memoized in :data:`embedding.relevance_cache`.

    Args:
        chunk: The retrieved chunk text.
        gold_spans: List of gold relevance strings for the question.

    Returns:
        ``True`` if any relevance criterion fires, ``False`` otherwise.
    """
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
