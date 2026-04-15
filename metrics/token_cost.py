"""Token cost metric.

Counts the total number of tokens the retrieved top-K chunks would occupy
if concatenated into an LLM prompt. Uses tiktoken with the encoding
configured in config.TIKTOKEN_ENCODING (cl100k_base by default, which is
the tokenizer used by gpt-4, gpt-4o and text-embedding-3-*).

Lower is better: a strategy that reaches the same retrieval quality with
fewer tokens lets the downstream LLM fit more context and costs less per
call.
"""

import tiktoken

from config import TIKTOKEN_ENCODING


_encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def token_cost(chunks: list) -> int:
    return sum(len(_encoder.encode(c)) for c in chunks)
