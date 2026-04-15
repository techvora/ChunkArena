# evaluation/chunk_stats.py

Per-collection chunk quality statistics. Independent of any query.

## Functions

- collection_redundancy(method, sample_size=REDUNDANCY_SAMPLE_SIZE)
    Draws a seeded random sample of chunks from
    all_chunks_cache[method] using random.Random with
    REDUNDANCY_SAMPLE_SEED from config. Embeds each sampled chunk via
    metrics.get_embedding, computes the full similarity matrix, and
    returns the mean of the upper triangle. Returns 0 when the sample
    has fewer than two chunks.

    Using a seeded random sample rather than [:sample_size] avoids
    the systematic bias where the early chunks of a document are
    often an abstract or a table of contents that are not
    representative of the rest of the corpus.

- chunk_stats(method)
    Computes num_chunks, avg_words, std_words, min_words, max_words,
    median_words, boundary_ratio (fraction of chunks ending on a
    sentence terminator), and calls collection_redundancy. Returns a
    dict that the runner turns into a row in stats_df and writes to
    chunk_stats.csv.

## Why it is independent of retrieval

These are properties of the chunk set itself, not of the retriever.
They let the report explain why a strategy wins or loses before any
retrieval happens: a strategy with very short chunks and very high
collection redundancy is going to struggle on Recall@K regardless of
the retriever.
