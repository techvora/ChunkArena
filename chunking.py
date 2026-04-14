import json
import re
import os
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import torch

# ------------------------------------------------------------
# 1. Initialize embedding model (BGE-M3)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. Reconstruct text from normalized atomic units
# ------------------------------------------------------------
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
    if start_idx == -1:
        # Fallback: approximate by length
        start_idx = 0
    end_idx = start_idx + len(chunk_text)

    overlapping_units = []
    for meta in metadata_map:
        if meta["start"] < end_idx and meta["end"] > start_idx:
            overlapping_units.append(meta["unit_id"])
    return {
        "char_start": start_idx,
        "char_end": end_idx,
        "unit_ids": overlapping_units,
        "unit_count": len(overlapping_units)
    }

# ------------------------------------------------------------
# 3. Chunking strategies
# ------------------------------------------------------------
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

def overlapping_chunk(units: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """Overlapping chunks using RecursiveCharacterTextSplitter."""
    # Same as fixed_size but with overlap > 0
    return fixed_size_chunk(units, chunk_size, overlap)

def sentence_chunk(units: List[Dict], sentences_per_chunk: int) -> List[Dict]:
    """Group sentences into chunks of ~N sentences each."""
    text, metadata_map = reconstruct_text_and_metadata(units)
    # Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_text = " ".join(sentences[i:i+sentences_per_chunk])
        chunk_meta = find_metadata_for_chunk(chunk_text, text, metadata_map)
        chunks.append({
            "chunk_id": f"sentence_{i//sentences_per_chunk}",
            "text": chunk_text,
            "metadata": chunk_meta
        })
    return chunks

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

def recursive_chunk(units: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """LangChain's RecursiveCharacterTextSplitter (already used in fixed_size)."""
    return fixed_size_chunk(units, chunk_size, overlap)

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

def semantic_chunk(units: List[Dict]) -> List[Dict]:
    """Use SemanticChunker from langchain-experimental."""
    text, metadata_map = reconstruct_text_and_metadata(units)
    docs = semantic_splitter.create_documents([text])
    result = []
    for i, doc in enumerate(docs):
        chunk_meta = find_metadata_for_chunk(doc.page_content, text, metadata_map)
        result.append({
            "chunk_id": f"semantic_{i}",
            "text": doc.page_content,
            "metadata": chunk_meta
        })
    return result

# ------------------------------------------------------------
# 4. Main processing function
# ------------------------------------------------------------
def chunk_normalized_documents(normalized_units: List[Dict], method: str, **kwargs) -> List[Dict]:
    """
    Apply a chunking method to the list of atomic units.
    Methods: fixed_size, overlapping, sentence, paragraph, recursive, header, semantic
    """
    if method == "fixed_size":
        return fixed_size_chunk(normalized_units, **kwargs)
    elif method == "overlapping":
        return overlapping_chunk(normalized_units, **kwargs)
    elif method == "sentence":
        return sentence_chunk(normalized_units, **kwargs)
    elif method == "paragraph":
        return paragraph_chunk(normalized_units)
    elif method == "recursive":
        return recursive_chunk(normalized_units, **kwargs)
    elif method == "header":
        return header_based_chunk(normalized_units)
    elif method == "semantic":
        return semantic_chunk(normalized_units)
    else:
        raise ValueError(f"Unknown method: {method}")

# ------------------------------------------------------------
# 5. Run all strategies and save outputs
# ------------------------------------------------------------
if __name__ == "__main__":
    # Paths
    NORMALIZED_FILE = "Banking_system_normalized.json"   # output from your normalize script
    OUTPUT_DIR = "created_chunks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load normalized atomic units
    with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
        units = json.load(f)

    # Define all methods to test
    methods = [
        "fixed_size",
        "overlapping",
        "sentence",
        "paragraph",
        "recursive",
        "header",
        "semantic"
    ]

    # Optional parameters for size/overlap
    chunk_size = 1000
    overlap = 200
    sentences_per_chunk = 3

    for method in methods:
        print(f"Processing {method}...")
        if method == "fixed_size":
            chunks = chunk_normalized_documents(units, method, chunk_size=chunk_size, overlap=0)
        elif method == "overlapping":
            chunks = chunk_normalized_documents(units, method, chunk_size=chunk_size, overlap=overlap)
        elif method == "sentence":
            chunks = chunk_normalized_documents(units, method, sentences_per_chunk=sentences_per_chunk)
        elif method == "recursive":
            chunks = chunk_normalized_documents(units, method, chunk_size=chunk_size, overlap=overlap)
        else:
            chunks = chunk_normalized_documents(units, method)

        output_file = os.path.join(OUTPUT_DIR, f"chunks_{method}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved {len(chunks)} chunks to {output_file}")

    print("\n✅ All chunking strategies completed.")