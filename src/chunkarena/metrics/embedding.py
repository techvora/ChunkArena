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

from chunkarena.config import DEVICE, EMBEDDING_MODEL, EMBEDDING_NORMALIZE


device = DEVICE
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

embedding_cache = {}
relevance_cache = {}


def get_embedding(text: str) -> np.ndarray:
    """Return the BGE-M3 dense embedding for a string, with memoization.

    The module owns a single :class:`SentenceTransformer` instance and a
    process-global :data:`embedding_cache` keyed by the raw text. Every
    component of the evaluation pipeline (relevance check, redundancy,
    RAG-quality proxies, dense retrieval) calls through here so that a
    string is embedded at most once per run regardless of how many
    metrics consume it.

    Args:
        text: Raw text to embed. Used verbatim as the cache key, so
            callers that want cache hits must pass identical strings.

    Returns:
        1-D numpy array of the normalized (when configured) embedding.
    """
    if text not in embedding_cache:
        embedding_cache[text] = embedder.encode(
            [text], convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_NORMALIZE,
        )[0]
    return embedding_cache[text]
