# ChunkArena

This repository benchmarks **chunking strategies** and **retrieval techniques** for a domain dataset (banking/finance), then produces an Excel report with retrieval-quality metrics and chunk-quality diagnostics.

## End-to-end flow (pipeline)

At a high level, the system goes through these stages:

1. **Extract documents** (PDF/DOCX → markdown-like text + provenance)
2. **Normalize** extracted text into **atomic units** (headings, paragraphs, tables, images, formulas)
3. **Chunk** normalized units using multiple chunking methods
4. **Index** chunks in **Qdrant** (one collection per chunking method)
5. **Evaluate** retrieval across methods + techniques using a **golden dataset**
6. **Export** results to `chunking_evaluation_full.xlsx`

## What to run (typical sequence)

### 0) Environment

Create/activate a virtual environment before running scripts.

If you already have the existing environment used in this repo:

```bash
source .chunk_bench/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 1) Extraction (documents → `extracted_docs.json`)

Script: `extract_text.py`

- **Purpose**: uses `docling` to convert `.pdf`/`.docx` into markdown and stores provenance (filesystem + internal metadata).
- **Output**: `extracted_docs.json`

Important: this script points at a `scrapped_data/` folder by default. If you use it, set `SOURCE_PATH` to your real input folder.

### 2) Normalization (raw extraction → `Banking_system_normalized.json`)

Script: `raw_2_normalize_json.py`

- **Input**: `Banking_system_extraction.json`
- **Output**: `Banking_system_normalized.json`
- **Idea**: convert the raw markdown-ish text into a clean list of atomic units:
  - `heading` (with heading level)
  - `paragraph`
  - `table` (markdown table captured as a block)
  - `image` (URLs / placeholders)
  - `formula` (LaTeX-like math, with a rule to avoid misclassifying currency)

Run:

```bash
python raw_2_normalize_json.py
```

### 3) Chunking (normalized units → `created_chunks/chunks_<method>.json`)

Script: `chunking.py`

- **Input**: `Banking_system_normalized.json`
- **Output directory**: `created_chunks/`
- **Chunking methods produced**:
  - `fixed_size`
  - `overlapping`
  - `sentence`
  - `paragraph`
  - `recursive`
  - `header`
  - `semantic`

Run:

```bash
python chunking.py
```

Notes on key design choices:
- **Metadata preservation**: `chunking.py` reconstructs a full text and builds a character-offset map so each chunk can carry metadata like covered unit IDs and approximate source offsets.
- **Semantic chunking**: uses LangChain’s `SemanticChunker` with BGE-M3 embeddings so chunk boundaries follow semantic breakpoints rather than fixed characters.

### 4) Indexing (chunks → Qdrant collections)

Script: `vector_db/qdrant_db.py`

- **Purpose**: embeds each chunk using `BAAI/bge-m3` and upserts it into Qdrant.
- **Index layout**: one Qdrant collection per chunking method (e.g., `fixed_size`, `semantic`, etc.).
- **Vector size**: 1024 (BGE-M3 dense embedding dimension).

Run:

```bash
python vector_db/qdrant_db.py
```

Prerequisite: Qdrant must be running on `localhost:6333` (as used in both indexing and evaluation).

### 5) Evaluation (retrieval benchmarking → `chunking_evaluation_full.xlsx`)

Script: `evaluate.py`

- **Golden dataset input**: `Golden_dataset/Banking_system.csv`
  - Expected columns: `Question`, `Answer`, `Facts`, `Topic`
- **Output**: `chunking_evaluation_full.xlsx`

Run:

```bash
python evaluate.py
```

## Evaluation: techniques used

For **each chunking method** (each Qdrant collection), `evaluate.py` tests multiple retrieval pipelines:

1. **Dense**
   - Qdrant vector search using the query embedding.
2. **Hybrid**
   - Dense retrieval + BM25 lexical scoring, fused by a weighted sum (linear fusion).
3. **Dense + Rerank**
   - Dense top-N retrieval, then rerank with a cross-encoder.
4. **Hybrid + Rerank**
   - Hybrid top-N retrieval, then rerank with a cross-encoder.

Models:
- **Embedder**: `SentenceTransformer("BAAI/bge-m3")`
- **Reranker**: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`

## Evaluation: how metrics are computed in this repo

### Short evaluation summary

`evaluate.py` evaluates retrieval quality by using `Golden_dataset/Banking_system.csv` as the source of **queries** (`Question`) and **ground truth** (`Answer + Facts`). For each chunking method, it queries the corresponding **Qdrant collection** and computes IR-style metrics based on a semantic relevance heuristic:

- A retrieved chunk is considered **relevant** if its cosine similarity to the ground truth text (BGE-M3 embeddings) is at least **0.65**.
- Retrieval techniques compared per chunking method: **Dense**, **Hybrid (Dense+BM25 fusion)**, **Dense + Rerank**, **Hybrid + Rerank**.
- Metrics computed per query then averaged: Recall@K/HitRate@K, Precision@K, MRR, nDCG@K, plus diagnostics like redundancy, semantic coherence, context completeness, boundary accuracy, and chunk entropy.
- Outputs are exported to `chunking_evaluation_full.xlsx` (one sheet per chunking method, plus thresholds and a best-per-metric summary).

### Core idea: “semantic relevance” labeling

This code does not rely on pre-labeled chunk IDs. Instead, it defines a chunk as “relevant” if it is **semantically similar** to the ground truth text.

1. Build ground truth text as:
   - `truth_text = (Answer + " " + Facts).lower()`
2. Compute embeddings for:
   - the retrieved chunk text
   - the ground truth text
3. Compute cosine similarity.
4. Mark relevant if similarity ≥ **0.65** (default threshold in `evaluate.py`).

That relevance test is then used to compute standard IR metrics.

### Retrieval-quality metrics (range: 0 to 1)

Computed using the relevance test above:

- **Recall@K**: 1 if any of top-K is relevant, else 0 (then averaged across queries).
- **Hit Rate@K**: identical to Recall@K in this code.
- **Precision@K**: (# relevant in top-K) / K.
- **MRR**: reciprocal rank of the first relevant result; 0 if none are relevant.
- **nDCG@K**: discounted gain with binary relevance, normalized by ideal DCG.

### Chunk-quality / context metrics

These help diagnose chunking behavior beyond “did we retrieve something relevant?”

- **Redundancy** (lower is better)
  - Average pairwise **Jaccard overlap** between the top-K chunks’ word sets.
  - Interprets “many near-duplicates in top-K” as undesirable.
- **Semantic coherence** (higher is better)
  - Average cosine similarity between the query embedding and each of the top-K chunk embeddings.
  - Useful for “are the retrieved chunks consistently about the query?”
- **Context completeness** (higher is better)
  - Ground truth is split into multiple “must-cover” sentences (Answer + comma-split Facts).
  - Measures the fraction of those sentences that are semantically covered by at least one retrieved chunk in top-K.
- **Boundary accuracy** (higher is better)
  - Fraction of top-K chunks that end with `.`, `!`, or `?`.
  - A proxy for clean chunk boundaries (heuristic).
- **Chunk entropy** (context-dependent; see below)
  - Shannon entropy of word distribution across the top-K chunks.
  - Higher often indicates more diverse information; very low can indicate repetitive boilerplate.

## What is a “good” score? (project thresholds)

`evaluate.py` writes a `Target_Thresholds` sheet into the output Excel file. These are the threshold bands used by this project:

| Metric | Poor | Acceptable | Good | Excellent |
|---|---:|---:|---:|---:|
| Recall@5 | <0.3 | 0.3–0.5 | 0.5–0.7 | >0.7 |
| Precision@5 | <0.2 | 0.2–0.4 | 0.4–0.6 | >0.6 |
| MRR | <0.3 | 0.3–0.5 | 0.5–0.7 | >0.7 |
| nDCG@5 | <0.3 | 0.3–0.5 | 0.5–0.7 | >0.7 |
| Hit Rate@5 | <0.4 | 0.4–0.6 | 0.6–0.8 | >0.8 |
| Redundancy (lower better) | >0.2 | 0.1–0.2 | 0.05–0.1 | <0.05 |
| Semantic coherence | <0.5 | 0.5–0.6 | 0.6–0.7 | >0.7 |
| Context completeness | <0.3 | 0.3–0.5 | 0.5–0.7 | >0.7 |
| Boundary accuracy | <0.5 | 0.5–0.7 | 0.7–0.9 | >0.9 |
| Chunk entropy | <5 | 5–7 | 7–9 | >9 |

Practical interpretation:
- If you care about **answering correctly**, prioritize **Recall@K / HitRate@K / MRR / nDCG**.
- If you care about **clean chunking behavior**, watch **Redundancy, Context completeness, Boundary accuracy**.
- Treat **Chunk entropy** as a diagnostic: “excellent > 9” is a project heuristic and may vary heavily by domain and chunk size.

## Output: what’s inside `chunking_evaluation_full.xlsx`

The evaluator writes:

- **One sheet per chunking method** (`fixed_size`, `semantic`, etc.)
  - Rows = retrieval techniques
  - Columns = metrics
- **`Target_Thresholds`**
  - The threshold table above
- **`Best_Per_Metric`**
  - For each method+metric, the single best technique and its value

## Repository notes / constraints

- Qdrant is assumed at `localhost:6333`.
- The evaluation script currently uses an **absolute path** for the golden CSV. If you move the repo, update `csv_path` in `evaluate.py` accordingly.
- Some scripts reference excluded folders (e.g., `scrapped_data/`) by default; adjust paths for your environment.