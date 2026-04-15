"""Boundary score.

Fraction of retrieved chunks that end on a sentence terminator (period,
exclamation, question mark, optionally followed by a closing quote). It is
a heuristic for how cleanly a strategy cuts text; higher is better but the
absolute threshold varies by domain.
"""

import re


def boundary_score(chunks: list) -> float:
    if not chunks:
        return 0.0
    return round(sum(1 for c in chunks if re.search(r"[.!?][\"']?\s*$", c.strip())) / len(chunks), 4)
