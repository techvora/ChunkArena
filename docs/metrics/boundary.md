# metrics/boundary.py

Boundary score. Fraction of retrieved chunks that end on a sentence
terminator.

## Formula

boundary(chunks) = count of chunks matching the regex
                    [.!?]["']?\\s*$ on the stripped chunk / len(chunks)

Returns 0 when chunks is empty.

## Interpretation

A diagnostic, not a quality metric. A high boundary score means the
chunker tends to respect sentence limits, which keeps the downstream
LLM from stitching sentence fragments across chunks. The absolute value
is informative only within a single domain; comparing across domains
is meaningless.
