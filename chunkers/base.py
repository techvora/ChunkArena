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

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading BGE-M3 on {device}...")
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
semantic_splitter = SemanticChunker(
    embed_model,
    breakpoint_threshold_type="percentile"
)


def reconstruct_text_and_metadata(units: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Convert list of atomic units into a single text string with markdown formatting.
    Also returns a list mapping each character position to original unit metadata.
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
    """
    Given a chunk substring, find which original units overlap with it.
    Returns aggregated metadata.
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
