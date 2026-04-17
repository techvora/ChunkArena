"""Stage 1 — Extraction pipeline (raw source documents → bronze JSON).

Walks a folder of source documents (PDF, DOCX) and runs docling's
DocumentConverter on each file to recover structured content along with
filesystem provenance (filename, absolute path, size, mtime). The result
is a list of dicts written as a single JSON artifact under
data/bronze/extracted/.

Downstream stages (normalization → chunking) consume the bronze JSON and
never re-read the original binary files, so every later step is
deterministic given the bronze layer.

Not a public API — the project-wide entrypoint is scripts/run_extraction.py,
which resolves paths from chunkarena.config and calls process_complex_folder.
"""

import os
import json
import time
from docling.document_converter import DocumentConverter
from chunkarena.config import SOURCE_FILE, EXTRACTED_DATA_OUTPUT_DIR

def process_single_file(file_path):
    """Extract structured content from a single PDF/DOCX file.

    Args:
        file_path: Absolute or relative path to the source document.

    Returns:
        Dict with ``content`` (markdown string) and ``provenance`` metadata.
    """
    converter = DocumentConverter()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    filename = os.path.basename(file_path)
    print(f"Processing: {filename}...")

    file_stats = os.stat(file_path)
    file_metadata = {
        "filename": filename,
        "file_path": os.path.abspath(file_path),
        "file_size_bytes": file_stats.st_size,
        "created_at": time.ctime(file_stats.st_ctime),
        "modified_at": time.ctime(file_stats.st_mtime),
        "extension": filename.split(".")[-1].lower()
    }

    result = converter.convert(file_path)
    md_content = result.document.export_to_markdown()

    internal_metadata = {}
    if hasattr(result.document, 'origin') and result.document.origin:
        internal_metadata = result.document.origin.dict()

    print(f"Done: {filename}")

    return {
        "content": md_content,
        "provenance": {
            "file_system": file_metadata,
            "document_internal": internal_metadata,
            "extraction_info": {
                "engine": "Docling",
                "pages": len(result.document.pages) if hasattr(result.document, 'pages') else "N/A"
            }
        }
    }


def process_complex_folder(folder_path):
    """Extract structured content from every PDF/DOCX file in a folder.

    Iterates the folder non-recursively, running docling's
    ``DocumentConverter`` on each ``.pdf`` / ``.docx`` file and pairing
    the extracted markdown with filesystem provenance (filename, absolute
    path, byte size, ctime, mtime, extension) plus any embedded document
    metadata docling exposes via ``result.document.origin``. Errors on
    individual files are logged and skipped so a single bad input does
    not abort the batch.

    Args:
        folder_path: Absolute or relative path to the directory
            containing source documents.

    Returns:
        List of dicts with ``content`` (markdown string) and
        ``provenance`` (nested dict with ``file_system``,
        ``document_internal`` and ``extraction_info`` sections). Empty
        list if ``folder_path`` does not exist.
    """
    converter = DocumentConverter()
    documents = []

    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} does not exist.")
        return documents

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".pdf", ".docx")):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}...")

            try:
                # 1. Capture File-System Metadata (Provenance)
                file_stats = os.stat(file_path)
                file_metadata = {
                    "filename": filename,
                    "file_path": os.path.abspath(file_path),
                    "file_size_bytes": file_stats.st_size,
                    "created_at": time.ctime(file_stats.st_ctime),
                    "modified_at": time.ctime(file_stats.st_mtime),
                    "extension": filename.split(".")[-1].lower()
                }

                # 2. Extract Content and Internal Metadata using Docling
                result = converter.convert(file_path)
                md_content = result.document.export_to_markdown()
                
                # 3. Get Embedded Document Metadata (Author, Title, etc.)
                # Docling stores this in the 'origin' or 'metadata' attributes
                internal_metadata = {}
                if hasattr(result.document, 'origin') and result.document.origin:
                    # Convert the internal metadata object to a dict
                    internal_metadata = result.document.origin.dict()

                # 4. Combine everything
                documents.append({
                    "content": md_content,
                    "provenance": {
                        "file_system": file_metadata,
                        "document_internal": internal_metadata,
                        "extraction_info": {
                            "engine": "Docling",
                            "pages": len(result.document.pages) if hasattr(result.document, 'pages') else "N/A"
                        }
                    }
                })
                print(f"Done: {filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    return documents

if __name__ == "__main__":
    SOURCE_PATH = SOURCE_FILE
    OUTPUT_FILE = EXTRACTED_DATA_OUTPUT_DIR

    docs = process_complex_folder(SOURCE_PATH)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)
    
    print(f"\nSaved {len(docs)} documents with full provenance to {OUTPUT_FILE}")