# chunkers/__init__.py

Registry and dispatch for the chunking package.

## What it exports

- Each individual strategy function: fixed_size_chunk, sentence_chunk,
  paragraph_chunk, header_based_chunk, semantic_chunk.
- CHUNKER_REGISTRY  A dict mapping the benchmark method name to the
  callable. overlapping and recursive both map to fixed_size_chunk. The
  runner distinguishes them by passing different overlap kwargs.
- chunk_normalized_documents(units, method, **kwargs)
    Looks up method in the registry and calls the strategy with the
    forwarded kwargs. Raises ValueError on an unknown method name.

## Why the registry

The top-level chunking.py entrypoint iterates CHUNK_METHODS from config
and dispatches through this one function. Adding a new strategy is a
two-step operation: implement the function in a new module, then add
the name to the registry and to config.CHUNK_METHODS.
