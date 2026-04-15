# metrics/precision.py

Precision@K. Fraction of the top-K chunks that are relevant.

## Formula

precision_at_k(chunks, gold_spans, k) = count(relevant in chunks[:k]) / k

Returns 0 when k is 0 to avoid a divide-by-zero.

## Interpretation

Precision answers the question, of the K chunks I show, how many are
actually useful. It is the right metric when the downstream LLM has a
small context window and irrelevant chunks actively hurt answer quality.
