# Retrieval latency

Wall-clock latency of the retrieval pipeline per (method, technique,
question). Measured directly in evaluation/runner.py because it is a
property of the retrieval call, not of the chunk set.

## Definition

Using time.perf_counter around the individual retrieval calls:

    t_dense       wall-clock of dense_search
    t_hybrid_bm25 wall-clock of hybrid_search (BM25+RRF portion only,
                  because the internal dense_search is cache-served
                  on this call)
    t_rerank_*    wall-clock of each rerank call

The reported per-technique latency composes these honestly:

    dense:         t_dense
    hybrid:        t_dense + t_hybrid_bm25
    dense_rerank:  t_dense + t_rerank_dense
    hybrid_rerank: t_dense + t_hybrid_bm25 + t_rerank_hybrid

Adding t_dense to the hybrid values is the correct accounting even
though dense_search is cache-served during the hybrid call in the
runner loop. In production, hybrid does pay the dense cost exactly
once per query, and the cache only exists because the runner
evaluates multiple techniques against the same query back to back.

## Interpretation

Lower is better. Latency is reported in summary.csv as
avg_latency_ms per (method, technique). It is not part of the
composite score because the weight depends on the deployment SLA
and should be chosen explicitly by the user.

## Interpretation

Lower is better. Latency is reported in summary.csv as
avg_latency_ms per (method, technique). It is not part of the
composite score because the weight depends on the deployment SLA and
should be chosen explicitly by the user.
