# evaluation/data_loader.py

Loads the golden dataset CSV and returns a list of question records.

## load_questions()

- Reads config.CSV_PATH with pandas.
- For each row, builds a dict with
    id         sequential integer, zero-indexed, unique per question.
    question   the Question column verbatim.
    gold_spans the Gold_spans column split on semicolons with each
               piece stripped of surrounding whitespace.
- Prints the load banner and the final count.
- Returns the list.

## Why this shape

The runner's main loop iterates over the list once per chunking
method. The integer id keeps per-question rows aligned across methods
and techniques in raw_results.csv, and also enables the paired t-test
in the significance table because the two samples it compares are
paired on question id.
