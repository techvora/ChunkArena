# metrics/recall.py

Recall@K over gold spans.

## Formula

recall_at_k(chunks, gold_spans, k) =
    count of gold spans that are individually covered by at least one
    chunk in chunks[:k] / len(gold_spans)

For each gold span the check is is_relevant(chunk, [span]), that is,
the relevance predicate is evaluated against a single-span list so that
every span is required to be covered, not just any span.

Returns 0 when gold_spans is empty.

## Interpretation

Recall answers, of the facts the answer needs, how many did the
retriever surface in the top K. For multi-fact questions this is the
metric that differentiates strategies that cover only part of the
answer from strategies that assemble the full picture.
