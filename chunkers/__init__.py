"""Chunker registry.

Exposes every individual strategy plus a single dispatch function,
chunk_normalized_documents, that the top-level chunking.py entrypoint uses
to iterate through every method.

Note on overlapping: it shares the splitter implementation with fixed_size
and only differs in the overlap parameter, so the registry aliases it to
fixed_size_chunk. recursive has its own module because it uses a richer
separator ladder and produces genuinely different chunks.
"""

from typing import List, Dict

from .fixed_size import fixed_size_chunk
from .recursive import recursive_chunk
from .sentence import sentence_chunk
from .paragraph import paragraph_chunk
from .header import header_based_chunk
from .semantic import semantic_chunk


CHUNKER_REGISTRY = {
    "fixed_size": fixed_size_chunk,
    "overlapping": fixed_size_chunk,
    "recursive": recursive_chunk,
    "sentence": sentence_chunk,
    "paragraph": paragraph_chunk,
    "header": header_based_chunk,
    "semantic": semantic_chunk,
}


def chunk_normalized_documents(normalized_units: List[Dict], method: str, **kwargs) -> List[Dict]:
    """
    Apply a chunking method to the list of atomic units.
    Methods: fixed_size, overlapping, sentence, paragraph, recursive, header, semantic
    """
    if method not in CHUNKER_REGISTRY:
        raise ValueError(f"Unknown method: {method}")
    return CHUNKER_REGISTRY[method](normalized_units, **kwargs)


__all__ = [
    "fixed_size_chunk",
    "recursive_chunk",
    "sentence_chunk",
    "paragraph_chunk",
    "header_based_chunk",
    "semantic_chunk",
    "chunk_normalized_documents",
    "CHUNKER_REGISTRY",
]
