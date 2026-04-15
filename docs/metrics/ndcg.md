# metrics/ndcg.py

Normalized Discounted Cumulative Gain at K with binary relevance.

## Formula

DCG  = sum over i in 0..k-1 of 1 / log2(i + 2)
       for positions where is_relevant(chunks[i], gold_spans) is True.

IDCG = sum over i in 0..min(len(gold_spans), k) - 1 of 1 / log2(i + 2)
       which is the DCG of the best possible ranking.

ndcg_at_k = DCG / IDCG,   or 0 when IDCG is 0.

## Interpretation

nDCG is the standard position-penalized IR metric. Ranking a relevant
chunk at position 1 contributes more to the score than ranking it at
position 5. Normalization by IDCG makes scores comparable across
questions with different numbers of gold spans.
