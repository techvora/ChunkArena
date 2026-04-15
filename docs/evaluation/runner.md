# evaluation/runner.py

Top-level orchestration for the evaluation pipeline. This is the
only file that touches every other module in the package.

## run() flow

1. load_questions from data_loader.
2. build_all_stores from chunk_store (Qdrant scroll plus BM25 build
   per method).
3. Compute chunk_stats per method and collect stats_df.
4. Main loop. For every (method, question):
    - get_embedding(query)
    - Time dense_search (t_dense_ms).
    - Time hybrid_search (t_hybrid_ms).
    - Time rerank on dense candidates (t_dense_rr_ms).
    - Time rerank on hybrid candidates (t_hybrid_rr_ms).
    - For each of dense, hybrid, dense_rerank, hybrid_rerank, trim
      to FINAL_K and score hit_at_k, mrr_score, precision_at_k,
      ndcg_at_k, recall_at_k, avg_rank_score, redundancy_score,
      boundary_score, token_cost, context_relevance, faithfulness,
      answer_correctness.
    - Append one raw_results row with the miss flag derived from
      avg_rank being NaN, plus the measured latency_ms.
5. Write raw_results.csv and chunk_stats.csv.
6. Aggregate summary_df by (method, technique). Averages include
   avg_token_cost, avg_latency_ms, context_relevance, faithfulness,
   answer_correctness. Append composite_score using
   COMPOSITE_WEIGHTS, rank by composite, add verdict columns via
   threshold_verdict.
7. Write summary.csv.
8. Print the overall best (method, technique) pair to stdout.
9. build_workbook to emit the 3-sheet benchmark_report.xlsx.

## Guarantees

- No metric logic lives in this file. Every formula is imported
  from metrics/.
- No retrieval logic lives in this file. Every technique is
  imported from evaluation/retrieval.py.
- Latency is measured where the retrieval calls are made because
  it is a property of the call, not a pure function of the chunk
  set.
- Significance testing and verdict sheet assembly were removed
  because the Experiment Matrix already communicates the result;
  reintroduce them only if the benchmark needs statistical proofs.
