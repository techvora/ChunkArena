"""Chunking entrypoint.

Loads the normalized atomic units and runs every registered chunking strategy,
writing one JSON file per method under CHUNK_OUTPUT_DIR. Only orchestration
lives here; each strategy implementation lives under chunkarena.chunking.strategies.
"""

import json
import os

from chunkarena.chunking import chunk_normalized_documents
from chunkarena.config import (
    NORMALIZED_FILE,
    CHUNK_OUTPUT_DIR,
    CHUNK_SIZE,
    OVERLAP,
    SENTENCES_PER_CHUNK,
    CHUNK_METHODS,
)


def main():
    os.makedirs(CHUNK_OUTPUT_DIR, exist_ok=True)

    with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
        units = json.load(f)

    for method in CHUNK_METHODS:
        print(f"Processing {method}...")
        if method == "fixed_size":
            chunks = chunk_normalized_documents(units, method, chunk_size=CHUNK_SIZE, overlap=0)
        elif method == "overlapping":
            chunks = chunk_normalized_documents(units, method, chunk_size=CHUNK_SIZE, overlap=OVERLAP, prefix="overlap")
        elif method == "sentence":
            chunks = chunk_normalized_documents(units, method, sentences_per_chunk=SENTENCES_PER_CHUNK)
        elif method == "recursive":
            chunks = chunk_normalized_documents(units, method, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        else:
            chunks = chunk_normalized_documents(units, method)

        output_file = os.path.join(CHUNK_OUTPUT_DIR, f"chunks_{method}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved {len(chunks)} chunks to {output_file}")

    print("\nAll chunking strategies completed.")


if __name__ == "__main__":
    main()
