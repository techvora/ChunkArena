"""Paragraph-level chunking.

The normalizer already produced atomic units tagged with a type. This strategy
simply keeps every unit whose type is paragraph and emits one chunk per unit,
preserving the original authoring boundaries without any splitter at all.
"""

from typing import List, Dict


def paragraph_chunk(units: List[Dict]) -> List[Dict]:
    """Emit one chunk per paragraph atomic unit, bypassing any splitter.

    The normalizer has already decomposed the source document into typed
    atomic units, so paragraph chunking is a pure filter: take every unit
    whose ``type == "paragraph"``, keep its original text verbatim, and
    tag it with its provenance id and position. Preserves authoring
    boundaries perfectly and incurs zero tokenization cost, at the
    expense of highly variable chunk lengths driven by the source style.

    Args:
        units: Normalized atomic units produced by the normalizer stage.

    Returns:
        List of chunk dicts with ``chunk_id``, ``text`` and a ``metadata``
        dict carrying ``unit_ids``, ``unit_type`` and source ``position``.
    """
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
