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

from chunkarena.config import HYBRID_CAND_K
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
    """Run a pure dense vector search against a method's Qdrant collection.

    Delegates to Qdrant's ``query_points`` with the precomputed query
    embedding and maps returned point ids to chunk texts through the
    in-memory id-to-text dict populated by
    :func:`chunk_store.build_all_stores`. Results are memoized in
    :data:`retrieval_cache` keyed by ``("dense", method, query_text,
    top_k)`` so repeated calls during the evaluation loop are free.

    Args:
        query_text: The natural-language query, used only as part of the
            cache key.
        query_emb: Precomputed dense embedding for ``query_text``.
        method: Chunking-method name (Qdrant collection name).
        top_k: Number of points to retrieve.

    Returns:
        Tuple ``(ids, texts)`` of parallel lists of length ``top_k``
        ordered by descending cosine similarity.
    """
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
    """Fuse dense vector search with BM25 via reciprocal rank fusion.

    Pulls the dense top-``HYBRID_CAND_K`` candidates (cache-served on
    second call), scores every chunk with the method's BM25 index on
    lowercased word tokens, takes the BM25 top-``HYBRID_CAND_K``, then
    combines the two ranked lists with standard RRF
    (``score = sum(1 / (60 + rank))`` across both sources). Returns the
    ``top_k`` highest-scoring ids. Results are memoized in
    :data:`retrieval_cache` keyed by ``("hybrid", method, query, top_k)``.

    Args:
        query: Natural-language query text.
        query_emb: Precomputed dense embedding for ``query``.
        method: Chunking-method name (Qdrant collection name).
        top_k: Number of fused results to return.

    Returns:
        Tuple ``(ids, texts)`` of the top-``top_k`` RRF-fused chunks.
    """
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
    """Rerank a candidate list with the cross-encoder and cache the result.

    Scores every ``(query, text)`` pair with the MS-MARCO MiniLM
    cross-encoder, sorts descending and returns the top ``top_k``. The
    ``tag`` argument is part of the cache key so that ``dense_rerank``
    and ``hybrid_rerank`` never collide: without it, two techniques that
    happen to produce overlapping candidate ids for the same query would
    read each other's cached scores. The key is also tagged with the
    first 20 candidate ids so changes to the candidate set invalidate
    correctly.

    Args:
        query: Natural-language query text.
        ids: Candidate chunk ids from the upstream retriever.
        texts: Chunk texts parallel to ``ids``.
        top_k: Number of reranked chunks to return.
        tag: Technique label (``"dense"`` or ``"hybrid"``) used to
            segregate cache entries.

    Returns:
        Tuple ``(ids, texts)`` of the top ``top_k`` reranked chunks. If
        ``texts`` is empty, returns two empty lists.
    """
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
