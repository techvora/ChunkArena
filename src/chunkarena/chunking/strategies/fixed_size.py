"""Fixed-size chunking with zero overlap by default.

Uses LangChain's RecursiveCharacterTextSplitter with a character budget. The
separator ladder tries paragraph break, then newline, then space, then raw
character split so that cuts happen at the cleanest available boundary that
still respects the fixed size budget.
"""

from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..base import reconstruct_text_and_metadata, find_metadata_for_chunk


def fixed_size_chunk(units: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """Split the document into character-budgeted chunks.

    Uses ``RecursiveCharacterTextSplitter`` with a simple separator ladder
    ``["\\n\\n", "\\n", " ", ""]`` so that cuts fall on the cleanest
    boundary that still fits the budget. This is the baseline strategy:
    cheap, deterministic, structure-agnostic, and well suited to
    homogeneous prose where paragraph boundaries are weak signals. When
    ``overlap`` is non-zero, the same function backs the ``overlapping``
    strategy row in the report.

    Args:
        units: Normalized atomic units produced by the normalizer stage.
        chunk_size: Maximum character length of an emitted chunk.
        overlap: Characters of overlap between consecutive chunks; set to
            zero for strict non-overlapping fixed-size chunking.

    Returns:
        List of chunk dicts with ``chunk_id``, ``text`` and the
        ``metadata`` dict returned by :func:`find_metadata_for_chunk`.
    """
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
