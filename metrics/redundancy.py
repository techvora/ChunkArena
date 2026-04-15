"""Redundancy score.

Standard corpus-free redundancy: mean pairwise cosine similarity among the
embeddings of the retrieved chunks. A value near 0 means the retrieved set
is semantically diverse, a value near 1 means the top-K are near duplicates
of each other. Lower is better.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import get_embedding


def redundancy_score(chunks: list) -> float:
    """Standard: mean pairwise cosine similarity among top-K chunks."""
    if len(chunks) < 2:
        return 0.0
    embs = np.array([get_embedding(c) for c in chunks])
    sim = cosine_similarity(embs)
    n = len(sim)
    triu_indices = np.triu_indices(n, k=1)
    sum_sim = np.sum(sim[triu_indices])
    num_pairs = n * (n - 1) / 2
    return round(sum_sim / num_pairs, 4)
