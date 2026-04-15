"""Retrieval techniques.

Implements the four techniques benchmarked per chunking method:

    dense          Qdrant vector search against the query embedding.
    hybrid         Dense top-N fused with BM25 top-N via reciprocal rank
                   fusion (k=60).
    dense_rerank   Dense top-N reranked by the cross-encoder.
    hybrid_rerank  Hybrid top-N reranked by the cross-encoder.

Retrieval and rerank results are memoized per (technique, method, query).
The rerank cache key is tagged with the source technique so dense_rerank and
hybrid_rerank never collide when called with overlapping candidate ids.
"""

from collections import defaultdict

import numpy as np

from config import HYBRID_CAND_K
from .models import client, cross_encoder
from .chunk_store import (
    all_chunks_cache,
    all_chunks_text_dict,
    bm25_models,
    word_re,
)


retrieval_cache = {}
rerank_cache = {}


def dense_search(query_text: str, query_emb: np.ndarray,
                 method: str, top_k: int):
    key = ("dense", method, query_text, top_k)
    if key in retrieval_cache:
        return retrieval_cache[key]
    res   = client.query_points(collection_name=method,
                                query=query_emb.tolist(), limit=top_k)
    ids   = [r.id for r in res.points]
    texts = [all_chunks_text_dict[method][i] for i in ids]
    retrieval_cache[key] = (ids, texts)
    return ids, texts


def hybrid_search(query: str, query_emb: np.ndarray,
                  method: str, top_k: int):
    key = ("hybrid", method, query, top_k)
    if key in retrieval_cache:
        return retrieval_cache[key]

    dense_ids, _ = dense_search(query, query_emb, method, HYBRID_CAND_K)

    bm25    = bm25_models[method]
    tokens  = word_re.findall(query.lower())
    scores  = bm25.get_scores(tokens)
    all_ids, _ = all_chunks_cache[method]
    bm25_ranked = sorted(zip(all_ids, scores),
                         key=lambda x: x[1], reverse=True)[:HYBRID_CAND_K]

    # Reciprocal Rank Fusion (standard RRF)
    fused = defaultdict(float)
    for i, cid in enumerate(dense_ids):
        fused[cid] += 1 / (60 + i)
    for i, (cid, _) in enumerate(bm25_ranked):
        fused[cid] += 1 / (60 + i)

    final_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:top_k]
    texts     = [all_chunks_text_dict[method][i] for i in final_ids]
    retrieval_cache[key] = (final_ids, texts)
    return final_ids, texts


def rerank(query: str, ids: list, texts: list,
           top_k: int, tag: str = "") -> tuple:
    """Rerank with cross-encoder; cache key includes tag to separate techniques."""
    if not texts:
        return [], []
    key = (query, tag, tuple(ids[:20]))
    if key in rerank_cache:
        s_ids, s_texts = rerank_cache[key]
        return s_ids[:top_k], s_texts[:top_k]

    pairs  = [[query, t] for t in texts]
    scores = cross_encoder.predict(pairs)
    idx    = np.argsort(scores)[::-1]
    s_ids  = [ids[i] for i in idx]
    s_texts = [texts[i] for i in idx]
    rerank_cache[key] = (s_ids, s_texts)
    return s_ids[:top_k], s_texts[:top_k]
