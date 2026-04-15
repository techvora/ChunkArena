"""Shared embedding model and caches.

All metric modules that need a vector (relevance, redundancy) and the
retrieval code in the evaluation package go through get_embedding here so
there is exactly one BGE-M3 instance in memory and one cache keyed by the
raw text.

Caches are module-level dicts so they persist for the full evaluation run.
The runner clears none of them; the intent is to memoize every string we
ever embed (queries, gold spans, retrieved chunks) across all methods and
techniques.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


device = "cpu"
embedder = SentenceTransformer("BAAI/bge-m3", device=device)

embedding_cache = {}
relevance_cache = {}


def get_embedding(text: str) -> np.ndarray:
    if text not in embedding_cache:
        embedding_cache[text] = embedder.encode([text], convert_to_numpy=True)[0]
    return embedding_cache[text]
