# metrics/mrr.py

Mean Reciprocal Rank. Per question, returns 1 / rank of the first
relevant chunk using 1-indexed positions, or 0 if no chunk is relevant.
Averaged across questions to give MRR.

## Formula

mrr(chunks, gold_spans) = 1 / i  for the smallest i where
                          is_relevant(chunks[i-1], gold_spans) is True,
                          else 0.

## Interpretation

MRR is a position-sensitive upgrade of Hit@K. A system that places the
correct chunk at rank 1 scores 1.0; at rank 2 scores 0.5; at rank 5
scores 0.2. It rewards rerankers that pull the right chunk to the top.
