# metrics/hit.py

Hit@K. Binary indicator per question: 1 if any of the top-K retrieved
chunks is relevant, else 0. Averaged across questions by the runner to
report the fraction of questions the system answers at all.

## Formula

hit_at_k(chunks, gold_spans, k) = int(any(is_relevant(c, gold_spans)
                                           for c in chunks[:k]))

## Interpretation

High Hit@K means the retriever is finding something relevant most of
the time. Low Hit@K is a hard upper bound on every other metric: if the
retriever does not surface a relevant chunk in the top K, reranking
cannot fix it.
