# evaluation/report_excel.py

Builds a focused 3-sheet benchmark_report.xlsx workbook using
openpyxl. The workbook contains only what is strictly needed to
compare chunking strategies across retrievers: a flat result matrix,
a per-query drilldown, and a visual heatmap.

## Top-level function

build_workbook(summary_df, raw_df, final_k, method_params=None)

Takes the summary and raw dataframes produced by the runner and
saves the styled workbook to BENCHMARK_XLSX. method_params comes
from config.METHOD_PARAMS and fills the Chunk Size and Overlap
columns on the Experiment Matrix sheet.

## Sheet 1: Experiment Matrix

One row per (chunking method, retriever technique) pair. Sorted by
technique then composite_score descending, so rows read as grouped
leaderboards per retriever. Columns match the project layout:

    Experiment ID, Retriever Method, Chunking Method,
    Chunk Size, Overlap (%), Top-K,
    Recall@K, Precision@K, MRR, Hit Rate,
    Context Relevance, Faithfulness, Answer Correctness,
    Token Cost, Latency (ms), Redundancy Score, Notes

- Experiment IDs are EXP-01, EXP-02, ... assigned after the sort.
- Chunk Size and Overlap (%) come from config.METHOD_PARAMS. Values
  may be strings like dynamic, var or ~3 sent for strategies where
  the splitter is not character-budgeted.
- Notes is auto-generated from the composite verdict plus tags for
  fast or slow latency (vs median), high token cost, and redundant
  top-K sets (redundancy >= 0.6).
- Conditional color scales: green-to-red for higher-is-better
  columns (Recall through Answer Correctness), red-to-green for
  lower-is-better columns (Token Cost, Latency, Redundancy).
- Freeze panes at column D so Experiment ID, Retriever and
  Chunking Method stay pinned while scrolling.

## Sheet 2: Raw Results

Per (method, technique, question) row with every metric the runner
computes, including the RAG-quality proxies and the token cost and
latency columns. Freeze-paned at column E so the question stays
pinned. This is the sheet for query-wise drilldown.

## Sheet 3: Heatmap

Per-method metric matrix filtered to the hybrid_rerank technique
(the strongest retriever in the benchmark). Includes Hit@K, MRR,
Precision@K, nDCG@K, Recall@K, Context Relevance, Faithfulness,
Answer Correctness, Redundancy and Composite. Conditional color
scales make the per-metric winners visually obvious without having
to read the numbers.

## Deliberately not in the workbook

- Cover or verdict page: the Experiment Matrix already ranks rows
  by composite_score within each technique.
- Separate Summary sheet: the Experiment Matrix is the summary.
- By Technique sheet: grouping by technique is already done in the
  Experiment Matrix sort order.
- Chunk Stats sheet: written to chunk_stats.csv for diagnostic use,
  not part of the primary comparison.
- Significance sheet: the paired t-test table is a statistical
  extra, not a comparison result.
- Metric Guide sheet: metric definitions live in docs/metrics/.
