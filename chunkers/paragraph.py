"""Paragraph-level chunking.

The normalizer already produced atomic units tagged with a type. This strategy
simply keeps every unit whose type is paragraph and emits one chunk per unit,
preserving the original authoring boundaries without any splitter at all.
"""

from typing import List, Dict


def paragraph_chunk(units: List[Dict]) -> List[Dict]:
    """Each paragraph (or atomic unit of type 'paragraph') becomes its own chunk."""
    chunks = []
    chunk_id = 0
    for unit in units:
        if unit.get("type") == "paragraph":
            chunks.append({
                "chunk_id": f"para_{chunk_id}",
                "text": unit.get("content", ""),
                "metadata": {
                    "unit_ids": [unit.get("id")],
                    "unit_type": "paragraph",
                    "position": unit.get("position")
                }
            })
            chunk_id += 1
    return chunks
