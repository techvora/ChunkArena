"""Semantic chunking.

Wraps LangChain Experimental's SemanticChunker, which embeds sentences with
BGE-M3 (loaded in base.py) and cuts chunks at embedding-distance breakpoints
using the percentile breakpoint strategy. Slowest of the strategies but tends
to give the cleanest topical boundaries.
"""

from typing import List, Dict

from .base import reconstruct_text_and_metadata, find_metadata_for_chunk, semantic_splitter


def semantic_chunk(units: List[Dict]) -> List[Dict]:
    """Use SemanticChunker from langchain-experimental."""
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
