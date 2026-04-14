import os
import json
import time
from docling.document_converter import DocumentConverter

def process_complex_folder(folder_path):
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
    SOURCE_PATH = "/home/root473/Documents/POC/ChunkArena/scrapped_data"
    OUTPUT_FILE = "extracted_docs.json"

    docs = process_complex_folder(SOURCE_PATH)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)
    
    print(f"\nSaved {len(docs)} documents with full provenance to {OUTPUT_FILE}")