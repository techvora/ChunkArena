# metrics/composite.py

Composite score helpers and per-metric verdict bands.

## threshold_verdict(metric, value)

Looks up the Good and Moderate bands for the given metric name in
config.THRESHOLDS and returns one of Good, Moderate, Bad or N/A.

For redundancy the direction is inverted because lower is better: a
value at or below good is Good, at or below moderate is Moderate,
otherwise Bad. For every other metric the direction is the usual
higher is better.

Returns N/A when the metric name is not in THRESHOLDS or the value is
NaN.

## Composite score

The composite score itself is assembled in evaluation.runner by
multiplying each column in config.COMPOSITE_WEIGHTS with the matching
summary column and summing. The result is appended to summary_df as
composite_score and used to rank the (method, technique) table in the
Excel report.
