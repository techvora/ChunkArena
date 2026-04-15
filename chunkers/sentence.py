"""Sentence-level chunking.

Splits the reconstructed text with nltk.tokenize.sent_tokenize, which uses
the Punkt unsupervised sentence tokenizer. This handles abbreviations,
decimals and common edge cases that the naive [.!?] regex mishandles. If
the Punkt data package is not present it is downloaded on first import.
Groups a configurable number of consecutive sentences into each chunk.
"""

from typing import List, Dict

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

from .base import reconstruct_text_and_metadata, find_metadata_for_chunk


def sentence_chunk(units: List[Dict], sentences_per_chunk: int) -> List[Dict]:
    """Group sentences into chunks of ~N sentences each."""
    text, metadata_map = reconstruct_text_and_metadata(units)
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_text = " ".join(sentences[i:i + sentences_per_chunk])
        chunk_meta = find_metadata_for_chunk(chunk_text, text, metadata_map)
        chunks.append({
            "chunk_id": f"sentence_{i // sentences_per_chunk}",
            "text": chunk_text,
            "metadata": chunk_meta
        })
    return chunks
