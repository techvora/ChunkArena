"""Shared chunking helpers.

Loads the BGE-M3 embedding model once (used by SemanticChunker) and exposes two
utilities that every chunking strategy relies on:

    reconstruct_text_and_metadata
        Rebuilds a single markdown-like string from the list of normalized
        atomic units and returns a character-offset map so downstream chunkers
        can recover which source units each chunk covers.

    find_metadata_for_chunk
        Given an emitted chunk substring, locates it in the reconstructed text
        and returns the overlapping unit ids plus offsets.
"""

from typing import List, Dict, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from chunkarena.config import (
    DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    SEMANTIC_BREAKPOINT_TYPE,
)


device = DEVICE
print(f"Loading {EMBEDDING_MODEL} on {device}...")
embed_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': EMBEDDING_NORMALIZE}
)
semantic_splitter = SemanticChunker(
    embed_model,
    breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
)


def reconstruct_text_and_metadata(units: List[Dict]) -> Tuple[str, List[Dict]]:
    """Flatten normalized atomic units into a markdown string and offset map.

    Every chunking strategy needs a single contiguous text to split, and
    also needs to know which source units each emitted chunk covers so
    downstream metrics and reporting can trace chunks back to provenance.
    This helper does both in one pass: it injects markdown heading markers,
    paragraph blank lines and bracketed placeholders for images/tables/
    formulas, then records the character span that each unit occupies in
    the reconstructed string.

    Args:
        units: Ordered list of normalized units. Each dict is expected to
            carry ``id``, ``type``, ``content``, ``position`` and, for
            headings, a numeric ``level``.

    Returns:
        A tuple ``(full_text, metadata_map)`` where ``full_text`` is the
        reconstructed markdown string and ``metadata_map`` is a list of
        dicts with keys ``start``, ``end``, ``unit_id``, ``unit_type``,
        ``unit_position`` and ``heading_level``.
    """
    text_parts = []
    metadata_map = []  # list of (start_char, end_char, unit_info)
    current_pos = 0

    for unit in units:
        unit_type = unit.get("type")
        content = unit.get("content", "")
        unit_id = unit.get("id")
        position = unit.get("position")
        level = unit.get("level")  # for headings

        if unit_type == "heading":
            # Markdown heading: '#' repeated level times
            prefix = "#" * level + " "
            formatted = prefix + content
        elif unit_type == "paragraph":
            formatted = content
        elif unit_type == "image":
            # Represent image as a text placeholder
            formatted = f"[Image: {content}]"
        elif unit_type == "table":
            formatted = f"[Table: {content}]"
        elif unit_type == "formula":
            formatted = f"[Formula: {content}]"
        else:
            formatted = content

        # Add a newline after each unit for clean separation
        formatted += "\n\n"
        start = current_pos
        end = start + len(formatted)
        metadata_map.append({
            "start": start,
            "end": end,
            "unit_id": unit_id,
            "unit_type": unit_type,
            "unit_position": position,
            "heading_level": level if unit_type == "heading" else None
        })
        text_parts.append(formatted)
        current_pos = end

    full_text = "".join(text_parts)
    return full_text, metadata_map


def find_metadata_for_chunk(chunk_text: str, full_text: str, metadata_map: List[Dict]) -> Dict:
    """Recover the source unit ids covered by an emitted chunk.

    Locates ``chunk_text`` in the reconstructed ``full_text`` via a first-
    occurrence search, then scans the offset map for every unit whose
    character span overlaps the chunk. If the chunk was produced by a
    splitter that normalized whitespace or tokens and cannot be found, the
    function returns an explicit failure record instead of silently
    attributing the chunk to unit zero, so broken metadata recovery is
    visible downstream.

    Args:
        chunk_text: The chunk substring to locate.
        full_text: The reconstructed document text produced by
            :func:`reconstruct_text_and_metadata`.
        metadata_map: The offset map produced alongside ``full_text``.

    Returns:
        Dict with ``char_start``, ``char_end``, ``unit_ids`` (list of
        overlapping unit ids), ``unit_count`` and
        ``metadata_recovery_failed`` (True when the chunk could not be
        located).
    """
    # Simple search: find the first occurrence (chunk may be exact)
    start_idx = full_text.find(chunk_text)
    recovery_failed = start_idx == -1
    if recovery_failed:
        # Splitter normalized whitespace or tokens. Do not silently
        # attribute the chunk to unit 0; flag the failure instead.
        return {
            "char_start": -1,
            "char_end": -1,
            "unit_ids": [],
            "unit_count": 0,
            "metadata_recovery_failed": True,
        }
    end_idx = start_idx + len(chunk_text)

    overlapping_units = []
    for meta in metadata_map:
        if meta["start"] < end_idx and meta["end"] > start_idx:
            overlapping_units.append(meta["unit_id"])
    return {
        "char_start": start_idx,
        "char_end": end_idx,
        "unit_ids": overlapping_units,
        "unit_count": len(overlapping_units),
        "metadata_recovery_failed": False,
    }
