"""RAG-quality proxy metrics.

Three embedding-based proxies for metrics that in their canonical
(RAGAS) form require an LLM judge or a generated answer:

    context_relevance(query, chunks)
        Canonical: LLM judges how relevant the retrieved context is to
        the query.
        Proxy:     mean cosine similarity between the query embedding
                   and each chunk embedding. Values in roughly 0 to 1
                   with higher meaning more relevant.

    faithfulness(chunks, gold_answer_text)
        Canonical: the system generates an answer and the judge checks
        that every claim in the answer is supported by the retrieved
        context.
        Proxy:     mean cosine similarity between each retrieved chunk
                   and the gold answer text. This asks whether the
                   retrieved context lives in the same semantic space
                   as the reference answer, which is a necessary but
                   not sufficient condition for a faithful generation.

    answer_correctness(chunks, gold_spans)
        Canonical: the generated answer is compared to a gold answer.
        Proxy:     for each gold span, take the maximum cosine
                   similarity between that span and any retrieved
                   chunk; average across spans. This is a retrieval-
                   side upper bound on correctness: if this number is
                   low, no downstream LLM can reconstruct the gold
                   answer from this context set.

All three are honest upper bounds or necessary conditions on true RAG
quality, not substitutes for LLM-as-judge metrics. Swap for the ragas
library driven by an LLM when you need generation-side numbers.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import get_embedding


def context_relevance(query: str, chunks: list) -> float:
    if not chunks or not query:
        return 0.0
    q    = get_embedding(query)
    embs = np.array([get_embedding(c) for c in chunks])
    sims = cosine_similarity([q], embs)[0]
    return round(float(np.mean(sims)), 4)


def faithfulness(chunks: list, gold_answer_text: str) -> float:
    if not chunks or not gold_answer_text:
        return 0.0
    a    = get_embedding(gold_answer_text)
    embs = np.array([get_embedding(c) for c in chunks])
    sims = cosine_similarity([a], embs)[0]
    return round(float(np.mean(sims)), 4)


def answer_correctness(chunks: list, gold_spans: list) -> float:
    if not chunks or not gold_spans:
        return 0.0
    chunk_embs = np.array([get_embedding(c) for c in chunks])
    scores = []
    for span in gold_spans:
        span_emb = get_embedding(span)
        sims     = cosine_similarity([span_emb], chunk_embs)[0]
        scores.append(float(np.max(sims)))
    return round(float(np.mean(scores)), 4)
