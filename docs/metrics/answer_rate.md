# Answer Rate

Fraction of questions for which the retriever surfaced at least one
relevant chunk in the top-K. Equivalent to 1 - miss_rate and
numerically identical to the mean of Hit@K when Hit@K is averaged
across questions with equal weight.

## Formula

    answer_rate = 1 - miss_rate

where miss_rate is the fraction of raw_results rows whose avg_rank
was NaN (no relevant chunk in the top-K).

## Why it is reported alongside Hit@K

Hit@K is the per-query binary indicator. Answer Rate is the dataset-
level summary of the same signal. Reporting both lets readers of the
Experiment Matrix and summary.csv speak about coverage in either
per-query or dataset-level terms without having to recompute.

## Where it appears

- Aggregated in summary.csv as the answer_rate column.
- Shown as the Answer Rate column on the Experiment Matrix sheet,
  positioned between Hit Rate and Context Relevance.

## Direction

Higher is better. A value of 1.0 means the retriever never misses;
a value of 0.5 means half the questions came back with no relevant
chunk in the top-K, which is a hard ceiling on every downstream
metric and an immediate red flag regardless of how high nDCG looks.
