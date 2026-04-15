"""Average rank of the first relevant chunk.

Returns the 1-indexed position of the first relevant retrieved chunk, or
NaN when no chunk is relevant. Averaged per method and technique by the
runner; a separate miss column tracks how often the NaN case fires so the
mean stays meaningful.
"""

import numpy as np

from .relevance import is_relevant


def avg_rank_score(chunks: list, gold_spans: list) -> float:
    for i, c in enumerate(chunks, 1):
        if is_relevant(c, gold_spans):
            return float(i)
    return np.nan
