"""Composite score and verdict mapping.

The composite score is a weighted linear combination of the main IR metrics
defined in config.COMPOSITE_WEIGHTS. threshold_verdict translates any single
metric value into a Good / Moderate / Bad label using the band definitions
in config.THRESHOLDS, inverting the direction for redundancy where lower is
better.
"""

import pandas as pd

from chunkarena.config import THRESHOLDS


def threshold_verdict(metric: str, value: float) -> str:
    """Map a raw metric value to a Good / Moderate / Bad label.

    Looks up the ``good`` / ``moderate`` band cutoffs for ``metric`` in
    :data:`config.THRESHOLDS` and returns the label the value falls
    into. For every standard IR metric higher is better, so the
    comparison runs ``>=``; for ``redundancy`` the direction is
    inverted (lower is better) so the comparison runs ``<=``. Missing
    metric keys or NaN values collapse to ``"N/A"`` so the caller can
    render a blank cell without a special case.

    Args:
        metric: Metric name; must match a key in
            :data:`config.THRESHOLDS`.
        value: Scalar metric value to classify.

    Returns:
        One of ``"Good"``, ``"Moderate"``, ``"Bad"`` or ``"N/A"``.
    """
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
