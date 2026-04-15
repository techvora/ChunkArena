import re
import json
import torch
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# 1. Initialize Embedding Model (BGE-M3)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Loading BGE-M3 on {device}...")
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Initialize Semantic Chunker
semantic_splitter = SemanticChunker(
    embed_model, 
    breakpoint_threshold_type="percentile" 
)

def get_metadata_bundle(doc):
    """Helper to extract provenance info for each chunk."""
    prov = doc.get("provenance", {})
    fs = prov.get("file_system", {})
    return {
        "source_file": fs.get("filename"),
        "source_path": fs.get("file_path"),
        "page_count": prov.get("extraction_info", {}).get("pages"),
        "extension": fs.get("extension")
    }

def markdown_hierarchy_chunk(text, doc_id, metadata):
    """Best for Docling output: Splits by # and ## while keeping headers in chunks."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)
    
    chunks = []
    for i, split in enumerate(md_header_splits):
        chunk_meta = metadata.copy()
        chunk_meta.update(split.metadata) # Adds the header name to metadata
        chunks.append({
            "chunk_id": f"{doc_id}_md_{i}",
            "text": split.page_content,
            "provenance": chunk_meta
        })
    return chunks

def fixed_size_chunk(text, doc_id, metadata, chunk_size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [{
        "chunk_id": f"{doc_id}_fix_{i}", 
        "text": c, 
        "provenance": metadata
    } for i, c in enumerate(chunks)]

def paragraph_chunk(text, doc_id, metadata):
    """Splits text by double newlines, commonly representing paragraph breaks."""
    # Split by double newline, filter out empty strings
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    return [{
        "chunk_id": f"{doc_id}_para_{i}", 
        "text": p, 
        "provenance": metadata
    } for i, p in enumerate(paragraphs)]

def sentence_chunk(text, doc_id, metadata):
    """Splits text into chunks of ~3-5 sentences each."""
    # Improved sentence splitting regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    temp_chunk = []
    
    for sent in sentences:
        temp_chunk.append(sent)
        # Groups sentences into chunks of 3 to provide better context
        if len(temp_chunk) >= 3:
            chunks.append(" ".join(temp_chunk))
            temp_chunk = []
            
    if temp_chunk: # Add remaining sentences
        chunks.append(" ".join(temp_chunk))
        
    return [{
        "chunk_id": f"{doc_id}_sent_{i}", 
        "text": c, 
        "provenance": metadata
    } for i, c in enumerate(chunks)]

def semantic_chunk(text, doc_id, metadata):
    docs = semantic_splitter.create_documents([text])
    return [{
        "chunk_id": f"{doc_id}_sem_{i}", 
        "text": d.page_content, 
        "provenance": metadata
    } for i, d in enumerate(docs)]

def chunk_all_documents(documents, chunk_method="markdown"):
    all_chunks = []
    for doc in documents:
        # Match your NEW JSON format
        doc_id = doc.get("provenance", {}).get("file_system", {}).get("filename", "unknown")
        text = doc.get("content", "")
        metadata = get_metadata_bundle(doc)
        
        if not text:
            continue

        if chunk_method == "markdown":
            chunks = markdown_hierarchy_chunk(text, doc_id, metadata)
        elif chunk_method == "fixed_size":
            chunks = fixed_size_chunk(text, doc_id, metadata)
        elif chunk_method == "semantic":
            chunks = semantic_chunk(text, doc_id, metadata)
        elif chunk_method == "paragraph":
            chunks = paragraph_chunk(text, doc_id, metadata)
        elif chunk_method == "sentence":
            chunks = sentence_chunk(text, doc_id, metadata)
        else:
            continue
            
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    INPUT_FILE = "/home/root473/Documents/POC/ChunkArena/extracted_docs.json"
    OUTPUT_DIR = "/home/root473/Documents/POC/ChunkArena/created_chunks"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with open(INPUT_FILE, "r") as f:
            docs = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        docs = []

    if docs:
        # Added 'markdown' as the primary method for Docling data
        methods = ["markdown", "fixed_size", "semantic", "paragraph", "sentence"]
        for method in methods:
            print(f"Processing method: {method}...")
            chunks = chunk_all_documents(docs, chunk_method=method)
            
            output_path = os.path.join(OUTPUT_DIR, f"chunks_{method}.json")
            with open(output_path, "w") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            print(f"Finished {method}: {len(chunks)} chunks saved.")