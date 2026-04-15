"""Fixed-size chunking with zero overlap by default.

Uses LangChain's RecursiveCharacterTextSplitter with a character budget. The
separator ladder tries paragraph break, then newline, then space, then raw
character split so that cuts happen at the cleanest available boundary that
still respects the fixed size budget.
"""

from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import reconstruct_text_and_metadata, find_metadata_for_chunk


def fixed_size_chunk(units: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """Fixed-size chunks with optional overlap (overlap=0 for no overlap)."""
    text, metadata_map = reconstruct_text_and_metadata(units)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_meta = find_metadata_for_chunk(chunk_text, text, metadata_map)
        result.append({
            "chunk_id": f"fixed_{i}",
            "text": chunk_text,
            "metadata": chunk_meta
        })
    return result
