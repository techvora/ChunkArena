"""Central configuration for ChunkArena.

All tunable paths, K values, model thresholds and benchmark weights live here so
that chunkers, metrics and the evaluation runner share a single source of truth.
"""

import os
from pathlib import Path

# Project root is two parents up from this file:
#   src/chunkarena/config.py  →  src/chunkarena  →  src  →  <PROJECT_ROOT>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# Models & Infrastructure (shared across every pipeline stage)
# ============================================================
# Every hardcoded model name, host, port and device setting used to live
# in 4+ files. Centralizing them here means a single edit switches the
# embedder/reranker/vector-store for the whole framework, and environment
# variables let you override without touching code.
# ------------------------------------------------------------

# ---- Compute device ----------------------------------------
# "auto" picks cuda when available, else cpu. Override with CHUNKARENA_DEVICE.
def _resolve_device(pref: str) -> str:
    """Resolve the torch device string for embedding and reranking models.

    Accepts an explicit preference (``"cuda"``, ``"cpu"``, ``"mps"``) and
    returns it unchanged, or resolves ``"auto"`` by probing torch for CUDA
    availability. Falls back to ``"cpu"`` if torch is missing or the probe
    raises, so importing this module never fails on a machine without GPU.

    Args:
        pref: Desired device string from env var ``CHUNKARENA_DEVICE`` or
            the hardcoded default ``"auto"``.

    Returns:
        A concrete device string safe to pass to HuggingFace / torch.
    """
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = _resolve_device(os.getenv("CHUNKARENA_DEVICE", "auto"))

# ---- Embedding model (dense retrieval + semantic chunking + metrics) ----
# Single source of truth. Swap this string to change the embedder everywhere.
EMBEDDING_MODEL     = os.getenv("CHUNKARENA_EMBED_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = int(os.getenv("CHUNKARENA_EMBED_DIM", "1024"))
EMBEDDING_NORMALIZE = True

# ---- Reranker (cross-encoder used in *_rerank techniques) ----
RERANKER_MODEL = os.getenv(
    "CHUNKARENA_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ---- Semantic chunker tuning ----
SEMANTIC_BREAKPOINT_TYPE = os.getenv("CHUNKARENA_SEMANTIC_BREAKPOINT", "percentile")

# ---- Vector store (Qdrant) ----
QDRANT_HOST = os.getenv("CHUNKARENA_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("CHUNKARENA_QDRANT_PORT", "6333"))

# ------------------------------------------------------------
# Stage 1 — Extraction (raw → bronze)
# ------------------------------------------------------------
SOURCE_FILE = str(DATA_DIR / "raw" / "Central_banks.pdf")
EXTRACTED_DATA_OUTPUT_DIR = str(DATA_DIR / "bronze" / "extracted")

# ------------------------------------------------------------
# Stage 2 — Normalization (bronze → silver)
# ------------------------------------------------------------
NORMALIZED_FILE = str(DATA_DIR / "silver" / "normalized" / "Central_banks.json")

# ------------------------------------------------------------
# Stage 3 — Chunking (silver → gold)
# ------------------------------------------------------------
CHUNK_OUTPUT_DIR = str(DATA_DIR / "gold" / "chunks" / "Central_banks")

# ------------------------------------------------------------
# Stage 4 — Indexing (gold chunks → Qdrant)
# ------------------------------------------------------------
CHUNKS_PATH = CHUNK_OUTPUT_DIR

# ---- Qdrant collection names for create new collection and use in evaluation ----
# Derived automatically: {dataset}_{method_prefix}_v{version}
# e.g. "central_banks_fixed_v1", "central_banks_semantic_v1"

COLLECTION_VERSION = int(os.getenv("CHUNKARENA_COLL_VERSION", "1"))

_DATASET_NAME = Path(SOURCE_FILE).stem.lower()

_METHOD_PREFIX = {
    "fixed_size":  "fixed",
    "overlapping": "overlap",
    "sentence":    "sent",
    "paragraph":   "para",
    "recursive":   "recur",
    "header":      "header",
    "semantic":    "semantic",
}

_COLLECTION_DEFAULTS = {
    method: f"{_DATASET_NAME}_{prefix}_v{COLLECTION_VERSION}"
    for method, prefix in _METHOD_PREFIX.items()
}
COLLECTION_NAMES = {
    method: os.getenv(f"CHUNKARENA_COLL_{method.upper()}", default)
    for method, default in _COLLECTION_DEFAULTS.items()
}

CHUNK_SIZE = 1000
OVERLAP = 300
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
# Stage 5 — Evaluation pipeline
# ------------------------------------------------------------
GOLDEN_DATASET_DIR = DATA_DIR / "gold" / "golden_dataset"
# Accepts .csv or .xlsx — the data_loader auto-detects the format.
# Override via env var CHUNKARENA_GOLDEN_DATASET (filename only, e.g. "Central_banks.xlsx").
_golden_file = os.getenv("CHUNKARENA_GOLDEN_DATASET", "Central_banks.xlsx")
GOLDEN_DATASET_PATH = str(GOLDEN_DATASET_DIR / _golden_file)
# Backward compat alias
CSV_PATH = GOLDEN_DATASET_PATH

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
# Stage 6 — Reporting output artifacts
# ------------------------------------------------------------
RESULTS_DIR = DATA_DIR / "results" / "current"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Include the golden dataset name in the report filename so results
# are traceable, e.g. "benchmark_report_Banking_system.xlsx"
_golden_stem = Path(_golden_file).stem
BENCHMARK_XLSX = str(RESULTS_DIR / f"benchmark_report_{_golden_stem}.xlsx")
