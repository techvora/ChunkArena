"""Central configuration for ChunkArena.

All tunable paths, K values, model thresholds and benchmark weights live here so
that chunkers, metrics and the evaluation runner share a single source of truth.
"""

import os


# ------------------------------------------------------------
# Data Extraction paths
# ------------------------------------------------------------
SOURCE_FILE = "scrapped_data/Banking_system.pdf"
EXTRACTED_DATA_OUTPUT_DIR = "data_extraction_from_source_file"

# ------------------------------------------------------------
# Qdrant indexing path
# ------------------------------------------------------------
CHUNKS_PATH = "created_chunks/banking_system"

# ------------------------------------------------------------
# Chunking pipeline
# ------------------------------------------------------------
NORMALIZED_FILE = "raw_to_norm._json_for_chunks/Banking_system_normalized.json"
CHUNK_OUTPUT_DIR = "created_chunks/banking_system"

CHUNK_SIZE = 1000
OVERLAP = 200
SENTENCES_PER_CHUNK = 3

# Tokenizer used for the token_cost metric. cl100k_base is the encoding
# shared by gpt-4, gpt-4o and text-embedding-3-*.
TIKTOKEN_ENCODING = "cl100k_base"

# Seed used by collection_redundancy when it samples chunks.
REDUNDANCY_SAMPLE_SEED = 42
REDUNDANCY_SAMPLE_SIZE = 300

# Per-method parameter display used by the Experiment Matrix report
# sheet. The numeric chunk_size and overlap_pct are shown verbatim in
# the Chunk Size and Overlap (%) columns. Strings are allowed for
# strategies where the splitter is not character-budgeted.
METHOD_PARAMS = {
    "fixed_size":  {"chunk_size": CHUNK_SIZE, "overlap_pct": "0%"},
    "overlapping": {"chunk_size": CHUNK_SIZE, "overlap_pct": f"{int(round(OVERLAP / CHUNK_SIZE * 100))}%"},
    "recursive":   {"chunk_size": CHUNK_SIZE, "overlap_pct": f"{int(round(OVERLAP / CHUNK_SIZE * 100))}%"},
    "sentence":    {"chunk_size": f"~{SENTENCES_PER_CHUNK} sent", "overlap_pct": "0%"},
    "paragraph":   {"chunk_size": "var", "overlap_pct": "0%"},
    "header":      {"chunk_size": "var", "overlap_pct": "0%"},
    "semantic":    {"chunk_size": "dynamic", "overlap_pct": "dynamic"},
}

# ------------------------------------------------------------
# Evaluation pipeline
# ------------------------------------------------------------
CSV_PATH = "/home/root473/Documents/POC/ChunkArena/Golden_dataset/Banking_system.csv"

FINAL_K = 5
RETRIEVAL_K = 50
HYBRID_CAND_K = 100

# Semantic relevance cutoff used by metrics.relevance.is_relevant. Values
# above this cosine similarity against any gold span mark the chunk as
# relevant. This number is calibrated per (embedder, domain) pair and the
# shipped default (0.72) is tuned for BGE-M3 on the banking corpus. For a
# different embedder or a different domain, override via the environment
# variable CHUNKARENA_SOFT_THRESHOLD or edit this constant directly.
# Recalibration procedure is documented in docs/metrics/relevance.md.
SOFT_THRESHOLD = float(os.getenv("CHUNKARENA_SOFT_THRESHOLD", "0.72"))

CHUNK_METHODS = [
    "fixed_size", "overlapping", "sentence",
    "paragraph", "recursive", "header", "semantic",
]

TECHNIQUES = ["dense", "hybrid", "dense_rerank", "hybrid_rerank"]

# ------------------------------------------------------------
# Composite scoring
# ------------------------------------------------------------
COMPOSITE_WEIGHTS = {
    "ndcg_at_k"     : 0.30,
    "mrr"           : 0.25,
    "hit_at_k"      : 0.20,
    "recall_at_k"   : 0.15,
    "precision_at_k": 0.10,
}

# The composite score is a weighted linear combination; weights must sum
# to exactly 1.0 or every downstream number is meaningless. Fail loudly at
# import time instead of silently producing a drifted ranking.
_composite_total = sum(COMPOSITE_WEIGHTS.values())
if abs(_composite_total - 1.0) > 1e-6:
    raise ValueError(
        f"COMPOSITE_WEIGHTS must sum to 1.0, got {_composite_total:.6f}. "
        f"Update config.COMPOSITE_WEIGHTS explicitly."
    )

THRESHOLDS = {
    "hit_at_k"      : {"good": 0.8,  "moderate": 0.5},
    "mrr"           : {"good": 0.8,  "moderate": 0.5},
    "precision_at_k": {"good": 0.7,  "moderate": 0.3},
    "ndcg_at_k"     : {"good": 0.7,  "moderate": 0.5},
    "recall_at_k"   : {"good": 0.8,  "moderate": 0.6},
    "redundancy"    : {"good": 0.3,  "moderate": 0.6},
    "composite_score": {"good": 0.75, "moderate": 0.65},
}

# ------------------------------------------------------------
# Output artifacts
# ------------------------------------------------------------
RAW_RESULTS_CSV = "raw_results.csv"
CHUNK_STATS_CSV = "chunk_stats.csv"
SUMMARY_CSV = "summary.csv"
BENCHMARK_XLSX = "benchmark_report.xlsx"
