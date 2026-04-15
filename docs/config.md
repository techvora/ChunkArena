# config.py

Central configuration module. Every other file in the project imports its
tunables from here so the benchmark stays controlled, which is the whole
point of this comparison: only the chunking method varies, and every other
variable is pinned to the same value across strategies.

## What is defined

Chunking pipeline constants
- NORMALIZED_FILE  Input JSON produced by the normalizer.
- CHUNK_OUTPUT_DIR Directory where per-method chunk JSONs are written.
- CHUNK_SIZE       Character budget for size-based strategies.
- OVERLAP          Character overlap for overlapping and recursive.
- SENTENCES_PER_CHUNK  Sentences grouped into each sentence chunk.

Evaluation pipeline constants
- CSV_PATH       Golden dataset path (columns Question and Gold_spans).
- FINAL_K        Top-K returned to the user and scored.
- RETRIEVAL_K    Candidates pulled from Qdrant before trimming to FINAL_K.
- HYBRID_CAND_K  Candidates pulled from each of dense and BM25 before RRF.
- SOFT_THRESHOLD Cosine similarity cutoff that declares a chunk relevant
                  to a gold span when substring match fails. Loaded from
                  the env var CHUNKARENA_SOFT_THRESHOLD if present, else
                  falls back to the hardcoded default. Recalibration
                  guidance lives in docs/metrics/relevance.md.
- CHUNK_METHODS  Ordered list of strategy names iterated by the runner.
- TECHNIQUES     Retrieval techniques compared per strategy.

Composite scoring
- COMPOSITE_WEIGHTS  Weights applied to the key metrics to build the single
                     score the benchmark ranks on. Must sum to 1.0; this
                     is asserted at config import time so a silent drift
                     fails loudly instead of producing a corrupted
                     ranking. The runner also logs the full dict at
                     startup so every run records its own weights.
- THRESHOLDS         Good and Moderate bands per metric. Used by the
                     composite module to emit verdicts in the Excel report.

Experiment Matrix display
- METHOD_PARAMS  Per-method chunk_size and overlap_pct display values
  used by the Experiment Matrix sheet. Values may be strings like
  dynamic, var or ~3 sent for strategies where the splitter is not
  character-budgeted.

Output artifacts
- RAW_RESULTS_CSV, CHUNK_STATS_CSV, SUMMARY_CSV, BENCHMARK_XLSX  Output
  file names at the project root.

## How it flows

1. chunking.py reads CHUNK_SIZE, OVERLAP, SENTENCES_PER_CHUNK,
   NORMALIZED_FILE, CHUNK_OUTPUT_DIR, CHUNK_METHODS.
2. evaluation.runner reads every evaluation constant and the output file
   names.
3. metrics.relevance reads SOFT_THRESHOLD.
4. metrics.composite reads THRESHOLDS.
5. evaluation.report_excel reads COMPOSITE_WEIGHTS and BENCHMARK_XLSX.

Changing a value here is the supported way to re-run the benchmark against
a different K, a different threshold, or a different weighting without
touching code.
