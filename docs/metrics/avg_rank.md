# metrics/avg_rank.py

Average rank of the first relevant chunk.

## Formula

avg_rank_score(chunks, gold_spans) = i where i is the 1-indexed
position of the first relevant chunk, else NaN.

The runner averages this across questions but also tracks a separate
miss counter (1 when NaN, else 0) so the reported mean rank is not
silently dragged up by miss cases.

## Interpretation

Lower is better. A value of 1 means the retriever is always returning
the right chunk at the very top. A value near K means the right chunk
is always present but buried. The miss_rate companion column in the
summary tells you how much of the population the mean actually covers.
