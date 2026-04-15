"""Semantic chunking.

Wraps LangChain Experimental's SemanticChunker, which embeds sentences with
BGE-M3 (loaded in base.py) and cuts chunks at embedding-distance breakpoints
using the percentile breakpoint strategy. Slowest of the strategies but tends
to give the cleanest topical boundaries.
"""

from typing import List, Dict

from ..base import reconstruct_text_and_metadata, find_metadata_for_chunk, semantic_splitter


def semantic_chunk(units: List[Dict]) -> List[Dict]:
    """Cut chunks at embedding-distance breakpoints using BGE-M3.

    Delegates to the module-level ``semantic_splitter`` (a LangChain
    Experimental ``SemanticChunker`` backed by BGE-M3 and the percentile
    breakpoint strategy configured in :mod:`chunkarena.config`). The
    splitter embeds each sentence, measures consecutive distances and
    cuts where the distance crosses the percentile threshold, producing
    topically coherent chunks of dynamic length. Slowest of the shipped
    strategies; best suited to long-form prose where topical drift is
    the dominant signal.

    Args:
        units: Normalized atomic units produced by the normalizer stage.

    Returns:
        List of chunk dicts with ``chunk_id``, ``text`` and recovered
        ``metadata``.
    """
    text, metadata_map = reconstruct_text_and_metadata(units)
    docs = semantic_splitter.create_documents([text])
    result = []
    for i, doc in enumerate(docs):
        chunk_meta = find_metadata_for_chunk(doc.page_content, text, metadata_map)
        result.append({
            "chunk_id": f"semantic_{i}",
            "text": doc.page_content,
            "metadata": chunk_meta
        })
    return result
