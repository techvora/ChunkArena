# Evaluation Logic Deep Dive (ChunkArena)

This document explains, step-by-step, how `evaluate.py` performs end-to-end evaluation using:

- the golden dataset in `Golden_dataset/Banking_system.csv`
- Qdrant vector DB collections (one per chunking method)

The goal is to produce a consistent, comparable benchmark across chunking strategies and retrieval techniques, and export results to `chunking_evaluation_full.xlsx`.

## 1) What evaluation is measuring (high-level goal)

Evaluation answers this question:

> Given a question from the golden dataset, can the retrieval pipeline fetch chunks from the indexed corpus that are semantically relevant to the known ground-truth answer/facts?

This is **retrieval evaluation**, not LLM answer generation evaluation. The outputs are retrieval-ranked chunks and their metric scores.

## 2) Inputs and prerequisites

### Golden dataset

File: `Golden_dataset/Banking_system.csv`

Expected columns used by `evaluate.py`:

- `Question`: the query text
- `Answer`: ground-truth answer text (may be empty)
- `Facts`: comma-separated facts (may be empty)
- `Topic`: optional metadata (loaded but not used for filtering in the current evaluation)

### Qdrant collections (vector DB)

`evaluate.py` assumes Qdrant is reachable at:

- host: `localhost`
- port: `6333`

It also assumes collections exist with names matching chunking methods:

- `fixed_size`
- `overlapping`
- `sentence`
- `paragraph`
- `recursive`
- `header`
- `semantic`

Each point in each collection is expected to have a payload containing:

- `text`: the chunk text string to evaluate

### Prerequisite pipeline steps

Before evaluation, you should have completed:

- Chunk generation: `chunking.py` → `created_chunks/chunks_<method>.json`
- Indexing: `vector_db/qdrant_db.py` → Qdrant collections populated with vectors and chunk text payloads

## 3) Models used during evaluation

`evaluate.py` loads two model types:

### A) Embedding model (dense retrieval + relevance checks)

- Model: `SentenceTransformer("BAAI/bge-m3")`
- Used for:
  - encoding queries
  - encoding chunk texts (for semantic metrics and relevance)
  - encoding ground truth text (for relevance)

Similarity used: **cosine similarity**.

### B) Cross-encoder reranker (reordering candidates)

- Model: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
- Used for:
  - scoring pairs \((query, chunk)\)
  - reranking candidate lists in the `... + Rerank` techniques

## 4) How the golden CSV becomes evaluation targets

For each CSV row, `evaluate.py` constructs:

### A) Query text

- `query = row["Question"]`

### B) Ground truth text (single combined target)

This is used for the main IR metrics (Recall/Precision/MRR/nDCG/HitRate):

- `truth_text = (Answer + " " + Facts).lower()`

Why combine Answer + Facts?
- It gives the evaluator one “semantic target” representing what the retrieval should support.

### C) Ground truth sentences list (coverage targets)

This is used for **Context Completeness**:

- If `Answer` exists: include Answer as one target sentence.
- If `Facts` exists: split by commas and include each non-empty fact as another target sentence.

Why split Facts?
- It lets the evaluator measure whether top-K retrieved chunks cover **multiple required pieces** (not just one).

## 5) How each chunking method maps to a Qdrant collection

Evaluation is run **per chunking method**, where:

- chunking method name == Qdrant collection name

This design is intentional:
- You can compare chunking strategies in isolation because each method has its own collection.

Within each method, the evaluator then compares different **retrieval techniques** (Dense/Hybrid/Rerank).

## 6) Retrieval techniques evaluated (what each one does)

For each query, the evaluator creates a ranked list of retrieved chunk texts using one of these pipelines:

### A) Dense

1. Encode query with BGE-M3 → query vector.
2. Query Qdrant using vector similarity (cosine).
3. Return top-N chunk texts.

This is the baseline semantic retrieval method.

### B) Hybrid (Dense + BM25 fusion)

Hybrid attempts to combine semantic similarity with lexical match.

Steps in this code:

1. Dense retrieval pulls a larger candidate set.
2. BM25 is built over that candidate set (tokenized).
3. BM25 scores are computed for the query against those candidates.
4. Dense scores and BM25 scores are each min-max normalized to \([0, 1]\).
5. Scores are fused:

\[
\text{combined} = \alpha \cdot \text{dense\_norm} + (1-\alpha)\cdot \text{bm25\_norm}
\]

Default in `evaluate.py`:
- `alpha = 0.5`

6. Candidates are sorted by `combined`, and the top-K are returned.

What hybrid is good for:
- entity names, rare terms, and exact keyword matches that dense embeddings might under-rank.

### C) Dense + Rerank

Two-stage pipeline:

1. Dense retrieval returns a larger candidate pool (e.g., top 50).
2. Cross-encoder scores each (query, candidate_chunk) pair.
3. Results are reordered by cross-encoder score, and top-K are returned.

This often improves “top of list” ranking metrics like MRR and nDCG.

### D) Hybrid + Rerank

Same two-stage idea, but candidates come from the hybrid retrieval step instead of pure dense retrieval.

## 7) The key evaluation trick: semantic relevance labeling

Most IR metrics require a binary label “relevant/not relevant” for each retrieved item. This project generates that label with a semantic similarity heuristic.

### Relevance function

For a retrieved chunk text `chunk_text` and a ground truth target `truth_text`:

1. Embed `chunk_text` with BGE-M3.
2. Embed `truth_text` with BGE-M3.
3. Compute cosine similarity.
4. Mark “relevant” if:

\[
\cos(\text{chunk}, \text{truth}) \ge 0.65
\]

Where 0.65 is the default threshold in `evaluate.py`.

### Why this matters

This choice controls the strictness of evaluation:

- Higher threshold → stricter relevance → lower measured Recall/Precision/MRR/nDCG
- Lower threshold → looser relevance → higher scores (risking inflated relevance)

## 8) Metrics computed (per query, then averaged)

For each query, each technique produces an ordered list of retrieved chunk texts. The evaluator computes:

### Retrieval-quality metrics (0 to 1)

These depend on semantic relevance labeling.

#### Recall@K

- Returns 1.0 if any of the top-K chunks is relevant, otherwise 0.0.
- Averaged across all queries.

#### HitRate@K

- Same as Recall@K in this code.

#### Precision@K

- \((\#\text{ relevant chunks in top-K}) / K\)

#### MRR (Mean Reciprocal Rank)

- Find the rank \(r\) of the first relevant chunk.
- Return \(1/r\).
- Return 0 if none are relevant.

#### nDCG@K

- Builds a binary relevance vector for top-K.
- Computes DCG using log discount and divides by ideal DCG (IDCG).

### Chunk/context diagnostic metrics

These help understand chunking behavior and retrieval list quality.

#### Redundancy@K (lower is better)

- For all pairs in the top-K chunks, compute Jaccard overlap between the sets of words.
- Return the average overlap.

Interpretation:
- High redundancy → top-K are near-duplicates → less useful context diversity.

#### Semantic coherence@K (higher is better)

- Embeds top-K chunks and compares each to the query embedding.
- Returns the average similarity.

Interpretation:
- High coherence → retrieved chunks are consistently “about” the query.

#### Context completeness@K (higher is better)

- Uses the ground truth sentences list (Answer + comma-split Facts).
- Checks whether each truth sentence is “covered” by any top-K chunk under the same semantic relevance threshold.
- Score is fraction covered.

Interpretation:
- Measures whether top-K contain all required pieces, not just one.

#### Boundary accuracy@K (higher is better)

- Fraction of top-K chunks ending with `.`, `!`, or `?`.

Interpretation:
- A heuristic proxy that chunk boundaries align with sentence-like endings.

#### Chunk entropy@K (context-dependent)

- Computes Shannon entropy of word distribution across the words in top-K chunks.

Interpretation:
- Higher can indicate more varied content; very low may indicate repetitive boilerplate.
- “Good” depends on domain and chunk size.

## 9) Aggregation: how per-query metrics become final scores

For each (chunking method, technique):

1. Compute each metric per query.
2. Average each metric across all queries.
3. Store a single average score per metric.

So the Excel output can be read as:

> “On average across all golden questions, technique X on chunk method Y achieved these metric values.”

## 10) Excel output: what gets written and how to interpret it

Output file: `chunking_evaluation_full.xlsx`

Sheets written:

### A) One sheet per chunking method

For example: `fixed_size`, `semantic`, `header`, etc.

- Rows: retrieval techniques (Dense / Hybrid / Dense + Rerank / Hybrid + Rerank)
- Columns: all metrics

### B) `Target_Thresholds`

The evaluator writes a quick rubric mapping metrics to:

- Poor
- Acceptable
- Good
- Excellent

Use this as an internal benchmark, not a universal standard.

### C) `Best_Per_Metric`

For each method and metric, the evaluator selects the technique with the highest value.

Important caution:
- Some metrics are “lower is better” (e.g., redundancy).
- `Best_Per_Metric` uses `max()` uniformly, so treat it as a convenience sheet, not the final truth.

## 11) Common pitfalls and how to avoid misreading results

- **Absolute path in `evaluate.py`**: golden CSV path is hardcoded as an absolute path; moving the repo requires updating it.
- **Relevance threshold sensitivity (0.65)**: if you change it, expect large shifts in most retrieval-quality metrics.
- **Hybrid BM25 scope**: BM25 is computed over candidates from dense retrieval (approximation), not the entire collection, which can reduce the visible impact of lexical scoring.
- **“Best” depends on your goal**:
  - If you care about “find anything relevant”: Recall@K/HitRate@K
  - If you care about ranking quality: MRR/nDCG
  - If you care about non-duplicative context: redundancy (lower), completeness (higher)
