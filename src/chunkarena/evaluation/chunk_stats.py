"""Per-collection chunk quality statistics.

Computes distribution metrics (word count stats, boundary ratio) and a
corpus-level redundancy number (mean pairwise cosine similarity over a
sample) for each method. These are independent of any query and describe
the chunk set itself, not the retrieval outcome.
"""

import random
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from chunkarena.config import REDUNDANCY_SAMPLE_SEED, REDUNDANCY_SAMPLE_SIZE
from chunkarena.metrics.embedding import get_embedding
from .chunk_store import all_chunks_cache


def collection_redundancy(method: str, sample_size: int = REDUNDANCY_SAMPLE_SIZE) -> float:
    """Estimate corpus-level redundancy for a chunking method.

    Draws a seeded random sample from the method's chunk set (seed comes
    from :data:`config.REDUNDANCY_SAMPLE_SEED` so runs are reproducible),
    embeds every sampled chunk, and returns the mean upper-triangle
    cosine similarity. A higher number means chunks repeat each other,
    which hurts retrieval diversity; the report inverts the verdict
    direction for this metric so lower values read as Good.

    Args:
        method: Chunking-method name (matches a key in
            :data:`chunk_store.all_chunks_cache`).
        sample_size: Upper bound on the number of chunks sampled. The
            actual sample is ``min(sample_size, len(chunks))``.

    Returns:
        Rounded mean pairwise cosine similarity, or ``0.0`` if fewer
        than two chunks are available.
    """
    _, texts = all_chunks_cache[method]
    rng      = random.Random(REDUNDANCY_SAMPLE_SEED)
    k        = min(sample_size, len(texts))
    sample   = rng.sample(texts, k) if k > 0 else []
    embs     = np.array([get_embedding(t) for t in sample])
    sim      = cosine_similarity(embs)
    n        = len(sim)
    triu = np.triu_indices(n, k=1)
    if n < 2:
        return 0.0
    sum_sim = np.sum(sim[triu])
    num_pairs = n * (n - 1) / 2
    return round(sum_sim / num_pairs, 4)


def chunk_stats(method: str) -> dict:
    """Compute query-independent structural stats for a chunking method.

    Summarises the shape of the emitted chunk set: count, word-count
    distribution (mean, std, min, max, median), the boundary ratio
    (fraction of chunks that end on a sentence-terminating punctuation
    mark) and the collection redundancy. These numbers describe the
    corpus itself and do not depend on any query or retrieval technique.

    Args:
        method: Chunking-method name present in
            :data:`chunk_store.all_chunks_cache`.

    Returns:
        Dict with keys ``method``, ``num_chunks``, ``avg_words``,
        ``std_words``, ``min_words``, ``max_words``, ``median_words``,
        ``boundary_ratio`` and ``collection_redundancy``.
    """
    _, texts  = all_chunks_cache[method]
    lengths   = [len(t.split()) for t in texts]
    boundary  = sum(1 for t in texts if re.search(r"[.!?][\"']?\s*$", t.strip()))
    return {
        "method"               : method,
        "num_chunks"           : len(texts),
        "avg_words"            : round(np.mean(lengths), 2),
        "std_words"            : round(np.std(lengths), 2),
        "min_words"            : int(np.min(lengths)),
        "max_words"            : int(np.max(lengths)),
        "median_words"         : round(float(np.median(lengths)), 2),
        "boundary_ratio"       : round(boundary / len(texts), 4),
        "collection_redundancy": collection_redundancy(method),
    }
