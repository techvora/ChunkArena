"""Recursive character chunking.

Distinct from fixed_size and overlapping by using a richer separator
hierarchy that the RecursiveCharacterTextSplitter walks top-down:

    paragraph break -> line break -> sentence terminators
    -> clause separators -> space -> raw character

This is the canonical configuration reported in benchmark literature as
recursive chunking and is genuinely different from the paragraph/newline/
space/raw ladder used by fixed_size and overlapping, so it earns its own
row in the report.
"""

from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import reconstruct_text_and_metadata, find_metadata_for_chunk


def recursive_chunk(units: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    text, metadata_map = reconstruct_text_and_metadata(units)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    chunks = splitter.split_text(text)
    result = []
    for i, chunk_text in enumerate(chunks):
        chunk_meta = find_metadata_for_chunk(chunk_text, text, metadata_map)
        result.append({
            "chunk_id": f"recursive_{i}",
            "text": chunk_text,
            "metadata": chunk_meta
        })
    return result
