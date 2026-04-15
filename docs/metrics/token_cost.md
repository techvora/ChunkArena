# metrics/token_cost.py

Token cost of the top-K retrieved chunks.

## Formula

token_cost(chunks) = sum of tiktoken encode length across chunks

The encoding is loaded once at module import time from
config.TIKTOKEN_ENCODING (cl100k_base by default, the tokenizer used by
gpt-4, gpt-4o and text-embedding-3-*).

## Interpretation

Lower is better. A strategy that matches another on nDCG but needs
fewer tokens to do so wins on cost and on downstream LLM latency
because more of the context window is free for the user prompt and
the answer.

The runner aggregates this per (method, technique) as avg_token_cost
in summary.csv. It is not in the composite score because the weight
depends on the deployment target and should be chosen explicitly by
the user.
