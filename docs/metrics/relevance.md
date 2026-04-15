# metrics/relevance.py

Defines whether a retrieved chunk is relevant to a question. Every IR
metric in this project is computed on top of this one predicate.

## Definition

is_relevant(chunk, gold_spans) returns True when any of the following
holds:

1. Substring match. Any gold span appears as a case-insensitive
   substring of the chunk. This is effectively the ground truth when
   the gold data is extractive.

2. Semantic match. The chunk embedding has cosine similarity at least
   SOFT_THRESHOLD (from config.py) against at least one gold span
   embedding. This is the fallback that makes paraphrased gold spans
   usable.

Both paths are memoized in relevance_cache keyed by (chunk,
tuple(gold_spans)), so the same pair is evaluated exactly once per run.

## Why two paths

A pure substring check is too strict for real data because gold spans
are often written in the user's words rather than copied verbatim.
A pure semantic check is too loose because any two banking paragraphs
look similar in embedding space. Combining them gives a conservative
bound that the runner can defend when reporting numbers.

## Recalibrating SOFT_THRESHOLD for a new embedder or domain

SOFT_THRESHOLD is calibrated per (embedder, domain) pair. The shipped
default 0.72 is tuned for BGE-M3 on the banking corpus. For any other
embedder or any other domain the number will drift and the reported
Hit, Recall and MRR will become optimistic or pessimistic.

Override without touching code:

    export CHUNKARENA_SOFT_THRESHOLD=0.68
    python evaluate.py

Or edit config.SOFT_THRESHOLD directly.

Manual recalibration procedure:

1. Take 20 to 50 questions from your gold dataset and label a small
   set of positive chunks (retrieved chunks that a human confirms
   contain the answer) and negative chunks (retrieved chunks that a
   human confirms do not).
2. For each labeled chunk, compute cosine similarity against the
   matching gold span using the same embedder the runner uses.
3. Plot or sort the two distributions. The target threshold is the
   value that maximizes separation: the lowest similarity observed on
   positives minus a small safety margin, typically 0.02 to 0.05.
4. Set the chosen value via CHUNKARENA_SOFT_THRESHOLD and rerun.

The runner prints the active SOFT_THRESHOLD at startup so every run's
effective value is captured in the log alongside the scores.
