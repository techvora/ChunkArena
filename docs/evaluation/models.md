# evaluation/models.py

Loads the retrieval and reranking models used across the evaluation
package.

## Exports

- cross_encoder  CrossEncoder cross-encoder/ms-marco-MiniLM-L-6-v2 on
  cpu. Used by retrieval.rerank to reorder candidate chunks by query
  relevance.
- client  QdrantClient pointed at localhost:6333. Used by
  retrieval.dense_search and chunk_store.get_all_chunks.

## Why it is separate from metrics/embedding.py

The embedder that scores relevance and builds query vectors is a
concern of the metrics package because it participates in the metric
definitions. The cross-encoder and the Qdrant client are concerns of
the evaluation package because they only participate in retrieval.
Keeping them apart means metric code never has to import retrieval
code and vice versa.
