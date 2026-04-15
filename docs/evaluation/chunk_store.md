# evaluation/chunk_store.py

Chunk store and BM25 index builder.

## State

- all_chunks_cache[method]     = (ids, texts) in Qdrant insertion order.
- all_chunks_text_dict[method] = dict mapping id to text for O(1)
                                 reverse lookup during retrieval.
- bm25_models[method]          = BM25Okapi over the tokenized texts.
- word_re                      = compiled regex \\w+ used both here and
                                 by retrieval.hybrid_search for
                                 tokenization.

## Functions

- get_all_chunks(collection_name)
    Scrolls the Qdrant collection in batches of 1000 until the server
    returns no next_offset, then returns parallel lists of ids and
    payload texts. Works for any collection size because scrolling is
    incremental.
- build_all_stores()
    For every method in CHUNK_METHODS, fills all three module-level
    caches and prints the chunk count. Called once by the runner at
    startup.
