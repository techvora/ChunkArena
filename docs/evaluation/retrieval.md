# evaluation/retrieval.py

Implements the four retrieval techniques compared per chunking method
and caches their outputs.

## Techniques

- dense_search(query_text, query_emb, method, top_k)
    Qdrant query_points against the per-method collection using the
    precomputed query embedding. Returns parallel lists of ids and
    texts.

- hybrid_search(query, query_emb, method, top_k)
    Runs dense retrieval for HYBRID_CAND_K candidates and BM25 scoring
    over the full chunk set for HYBRID_CAND_K candidates, then fuses
    the two with Reciprocal Rank Fusion at k=60:
        fused[id] += 1 / (60 + rank_in_list)
    Final top_k is the argmax over the fused score.

- rerank(query, ids, texts, top_k, tag)
    Cross-encoder predict over (query, text) pairs, argsort desc, take
    top_k. The tag parameter is part of the cache key so dense_rerank
    and hybrid_rerank never collide even when they happen to see the
    same initial 20 ids.

## Caches

- retrieval_cache  keyed by (technique, method, query, top_k).
- rerank_cache     keyed by (query, tag, tuple(ids[:20])).

Both are module-level and never cleared for the duration of a run.
