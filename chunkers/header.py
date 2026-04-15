"""Header-aware chunking.

Uses LangChain's MarkdownHeaderTextSplitter. The reconstructed text already has
markdown heading prefixes injected by base.reconstruct_text_and_metadata, so
the splitter can group each section under its nearest H1/H2/H3 header and emit
one chunk per structural section.
"""

from typing import List, Dict

from langchain_text_splitters import MarkdownHeaderTextSplitter

from .base import reconstruct_text_and_metadata, find_metadata_for_chunk


def header_based_chunk(units: List[Dict]) -> List[Dict]:
    """
    Use MarkdownHeaderTextSplitter to split by headings.
    Requires reconstructing markdown text with proper heading levels.
    """
    text, metadata_map = reconstruct_text_and_metadata(units)
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    splits = splitter.split_text(text)
    result = []
    for i, split in enumerate(splits):
        chunk_meta = find_metadata_for_chunk(split.page_content, text, metadata_map)
        # Also add header info from split.metadata
        chunk_meta["headers"] = split.metadata
        result.append({
            "chunk_id": f"header_{i}",
            "text": split.page_content,
            "metadata": chunk_meta
        })
    return result
