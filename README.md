# ChunkArena

A controlled-variable benchmark of seven chunking strategies on the same
corpus, the same retriever, the same reranker and the same golden
dataset. Only the chunking method varies.

## Chunking strategies compared

1. fixed_size         RecursiveCharacterTextSplitter with overlap = 0
2. overlapping        RecursiveCharacterTextSplitter with overlap > 0
3. recursive          RecursiveCharacterTextSplitter (canonical config)
4. sentence           Fixed number of sentences per chunk
5. paragraph          One chunk per normalized paragraph unit
6. header             MarkdownHeaderTextSplitter over H1, H2, H3
7. semantic           langchain_experimental SemanticChunker with BGE-M3

overlapping reuses the fixed_size splitter with a non-zero overlap.
recursive has its own module with a richer separator ladder
(sentence and clause terminators before space and raw character) and
produces genuinely different chunks from fixed_size and overlapping.

## Retrieval techniques compared per strategy

- dense          Qdrant vector search with BGE-M3
- hybrid         Dense + BM25 via Reciprocal Rank Fusion (k = 60)
- dense_rerank   Dense top-N reranked by ms-marco MiniLM cross-encoder
- hybrid_rerank  Hybrid top-N reranked by the same cross-encoder

## Metrics computed

- Hit@K
- MRR
- Precision@K
- nDCG@K
- Recall@K (span-level coverage: fraction of gold spans individually
  covered by at least one chunk in top-K)
- Avg Rank (with miss_rate companion)
- Answer Rate (1 - miss_rate; dataset-level coverage)
- Redundancy (mean pairwise cosine similarity in top-K)
- Diversity (1 - Redundancy)
- Boundary ratio
- Token cost (tiktoken cl100k_base, sum over top-K per query, averaged)
- Latency (wall-clock ms per retrieval call, averaged)
- Context Relevance (embedding proxy: mean cosine query-to-chunks)
- Faithfulness (embedding proxy: mean cosine chunks-to-gold-answer)
- Answer Correctness (embedding proxy: mean of max cosine per gold
  span against retrieved chunks)
- Composite score (weighted combination of nDCG, MRR, Hit, Recall,
  Precision). Token cost, latency and the RAG-quality proxies are
  reported alongside but are not in the composite because their
  weight depends on the deployment target.
- Collection-level chunk stats (count, word distribution, seeded
  random-sample collection redundancy)

Context Relevance, Faithfulness and Answer Correctness are embedding
proxies for the canonical RAGAS metrics. They are honest retrieval-
side upper bounds, not substitutes for LLM-judge generation metrics.
See docs/metrics/rag_quality.md.

## Repository layout

```
config.py                 All tunables in one place.
chunkers/
  base.py                 Text reconstruction, metadata map, BGE-M3
                          load, SemanticChunker.
  fixed_size.py           RecursiveCharacterTextSplitter (overlap=0).
  recursive.py            RecursiveCharacterTextSplitter with a rich
                          separator ladder (sentence and clause aware).
  sentence.py             Sentence-grouped chunking via nltk Punkt.
  paragraph.py            One chunk per paragraph unit.
  header.py               MarkdownHeaderTextSplitter strategy.
  semantic.py             SemanticChunker strategy.
  __init__.py             Registry and dispatch.
metrics/
  embedding.py            Shared embedder and caches.
  relevance.py            is_relevant predicate.
  hit.py  mrr.py  precision.py  ndcg.py  recall.py  avg_rank.py
  redundancy.py  boundary.py  token_cost.py  composite.py
  rag_quality.py          Context Relevance, Faithfulness, Answer
                          Correctness (embedding proxies).
  __init__.py             Flat re-export.
evaluation/
  models.py               Cross-encoder and Qdrant client.
  data_loader.py          Golden dataset loader.
  chunk_store.py          Qdrant scroll plus BM25 index.
  chunk_stats.py          Per-collection stats.
  retrieval.py            dense, hybrid, rerank.
  report_excel.py         8-sheet styled workbook.
  runner.py               Orchestration.
  __init__.py             Exposes run.
chunking.py               Entrypoint: runs every chunker.
evaluate.py               Entrypoint: runs the evaluation.
docs/                     Per-file documentation mirroring the tree.
created_chunks/           Output of chunking.py.
Golden_dataset/           Input CSV for evaluate.py.
vector_db/                Qdrant indexer script.
```

Each file has a dedicated doc under docs/ with the same path: for
example metrics/ndcg.py is documented at docs/metrics/ndcg.md.

## End-to-end pipeline

1. Extraction   extract_text.py converts PDFs and DOCX to markdown.
2. Normalize    raw_2_normalize_json.py produces atomic units.
3. Chunk        python chunking.py writes one JSON per method under
                created_chunks/.
4. Index        python vector_db/qdrant_db.py creates one Qdrant
                collection per method at localhost:6333.
5. Evaluate     python evaluate.py runs the benchmark and writes the
                four output artifacts.

## Output artifacts

- raw_results.csv        per (question, method, technique) row
- chunk_stats.csv        per method chunk quality snapshot
- summary.csv            aggregated and ranked scores with verdicts
- benchmark_report.xlsx  3-sheet focused report:
    1. Experiment Matrix  one row per (method, technique) with
                          Experiment ID, Retriever, Chunker,
                          Chunk Size, Overlap, Top-K, Recall,
                          Precision, MRR, Hit Rate, Context
                          Relevance, Faithfulness, Answer
                          Correctness, Token Cost, Latency,
                          Redundancy, Notes
    2. Raw Results        per-question drilldown with every metric
    3. Heatmap            per-method visual matrix filtered to
                          hybrid_rerank

## Configuration

Every tunable lives in config.py: paths, K values, thresholds,
composite weights, method list. Change a value there and both scripts
pick it up.

## How to add a new chunker

1. Create chunkers/mymethod.py with a function that takes units and
   returns a list of chunk dicts.
2. Import it in chunkers/__init__.py and add it to CHUNKER_REGISTRY.
3. Add the name to config.CHUNK_METHODS.
4. If the chunker takes parameters, add a branch in chunking.py.
5. Write docs/chunkers/mymethod.md.

## How to add a new metric

1. Create metrics/mymetric.py with a pure function.
2. Import and re-export it from metrics/__init__.py.
3. Wire it into the main loop in evaluation/runner.py and add a
   column to the summary aggregation if needed.
4. Write docs/metrics/mymetric.md.

## Prerequisites

- Qdrant running at localhost:6333.
- BGE-M3 and ms-marco MiniLM cross-encoder downloadable by
  sentence-transformers.
- Dependencies in requirements.txt.
