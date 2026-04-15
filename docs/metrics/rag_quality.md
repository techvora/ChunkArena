# metrics/rag_quality.py

Three embedding-based proxy metrics for RAG quality. Each has a
canonical RAGAS definition that requires an LLM judge or a generated
answer; the implementations here stay in the embedding space so the
benchmark runs without an LLM endpoint. Swap for true RAGAS when you
wire in an LLM key.

## context_relevance(query, chunks)

Canonical: an LLM judges how relevant the retrieved context is to the
query.

Proxy: mean cosine similarity between the query embedding and each
retrieved chunk embedding. Value range is roughly 0 to 1 with higher
meaning more relevant. Returns 0 for empty input.

Interpretation: a strategy with high context relevance retrieves
chunks that live close to the query in semantic space. This is a
necessary condition for good RAG but not sufficient, because similar
text is not the same as text that answers the question.

## faithfulness(chunks, gold_answer_text)

Canonical: the system generates an answer and an LLM judge checks
that every claim in the answer is supported by the retrieved context.

Proxy: mean cosine similarity between each retrieved chunk and the
gold answer text. If the gold answer is not present in the CSV the
runner uses the joined gold spans instead (see data_loader). Returns
0 for empty input.

Interpretation: a strategy with high proxy faithfulness surfaces
context that lives in the same semantic space as the reference
answer, which is the embedding-side upper bound on an LLM being able
to generate a faithful answer from that context.

## answer_correctness(chunks, gold_spans)

Canonical: a generated answer is compared to a gold answer by an LLM
judge.

Proxy: for each gold span, take the max cosine similarity between
that span and any retrieved chunk, then average across spans. Returns
0 for empty input.

Interpretation: this is a retrieval-side upper bound on correctness.
If this number is low, no downstream LLM can reconstruct the gold
answer from the retrieved context, regardless of generation quality.

## Why these are proxies, not substitutes

Embedding similarity answers did we retrieve text that looks like the
answer. The real RAGAS metrics answer did the LLM actually produce a
faithful and correct answer. Those two questions overlap but do not
match: you can retrieve perfect context and an LLM can still
hallucinate, and you can retrieve mediocre context and an LLM can
still stitch together a correct answer. These proxies are honest
indicators of the retrieval-side ceiling on RAG quality and are
labeled as proxies in the report sheet.
