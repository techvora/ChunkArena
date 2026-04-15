"""nDCG@K with binary relevance.

DCG is the sum of 1 / log2(rank + 2) across positions where the retrieved
chunk is relevant. IDCG is the DCG of the best possible ranking given
min(len(gold_spans), K) relevant items at the top. The returned value is
DCG / IDCG, or 0 when there are no gold spans at all.
"""

import numpy as np

from .relevance import is_relevant


def ndcg_at_k(chunks: list, gold_spans: list, k: int) -> float:
    dcg = sum(1 / np.log2(i + 2)
              for i, c in enumerate(chunks[:k])
              if is_relevant(c, gold_spans))
    idcg = sum(1 / np.log2(i + 2)
               for i in range(min(len(gold_spans), k)))
    return dcg / idcg if idcg > 0 else 0.0
