# metrics/redundancy.py

Redundancy of a retrieved top-K set.

## Formula

redundancy(chunks) = mean pairwise cosine similarity among the
embeddings of the K chunks, computed over the upper triangle of the
similarity matrix.

Returns 0 when there are fewer than two chunks.

## Interpretation

Lower is better. A value near 0 means the K chunks cover diverse facts
and the downstream LLM has unique context to work with. A value near
1 means the retriever is returning K near-duplicates, which wastes
context window and masks low Recall behind a high Hit@K.

The runner also reports diversity = 1 - redundancy for readability.
