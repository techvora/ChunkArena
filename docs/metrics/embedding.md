# metrics/embedding.py

Shared embedding model and caches for every metric and retrieval call.

## Exports

- embedder  SentenceTransformer BAAI/bge-m3 on cpu. Module-level so it
  loads exactly once for the whole run.
- embedding_cache  dict[str, ndarray] keyed by raw text. Every string
  that is ever embedded during evaluation ends up here.
- relevance_cache  dict[(chunk, tuple(gold_spans)), bool] used by
  relevance.is_relevant to skip duplicate work.
- get_embedding(text)  Returns a numpy vector, populating the cache on
  first call.

## Why caches are module-level

The benchmark reuses the same queries and the same retrieved chunks
across many (method, technique) combinations. Module-level caches let
every downstream metric reuse the same vector without having to thread
a cache object through function signatures.
