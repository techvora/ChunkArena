"""Composite score and verdict mapping.

The composite score is a weighted linear combination of the main IR metrics
defined in config.COMPOSITE_WEIGHTS. threshold_verdict translates any single
metric value into a Good / Moderate / Bad label using the band definitions
in config.THRESHOLDS, inverting the direction for redundancy where lower is
better.
"""

import pandas as pd

from config import THRESHOLDS


def threshold_verdict(metric: str, value: float) -> str:
    if metric not in THRESHOLDS or pd.isna(value):
        return "N/A"
    t = THRESHOLDS[metric]
    if metric == "redundancy":   # lower is better
        if value <= t["good"]:
            return "Good"
        elif value <= t["moderate"]:
            return "Moderate"
        else:
            return "Bad"
    else:                         # higher is better
        if value >= t["good"]:
            return "Good"
        elif value >= t["moderate"]:
            return "Moderate"
        else:
            return "Bad"
