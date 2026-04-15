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

from config import REDUNDANCY_SAMPLE_SEED, REDUNDANCY_SAMPLE_SIZE
from metrics.embedding import get_embedding
from .chunk_store import all_chunks_cache


def collection_redundancy(method: str, sample_size: int = REDUNDANCY_SAMPLE_SIZE) -> float:
    """
    Standard corpus-level redundancy: mean pairwise cosine similarity
    over a seeded random sample of chunks.
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
