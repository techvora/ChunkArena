"""Extraction entrypoint.

Thin wrapper around chunkarena.extraction.pipeline. Extracts text from every
document under data/raw and writes structured JSON to data/bronze/extracted.
"""

import json
import os

from chunkarena.extraction.pipeline import process_single_file
from chunkarena.config import SOURCE_FILE, EXTRACTED_DATA_OUTPUT_DIR


if __name__ == "__main__":
    os.makedirs(EXTRACTED_DATA_OUTPUT_DIR, exist_ok=True)

    doc = process_single_file(SOURCE_FILE)

    # Derive output filename from the source PDF name
    base_name = os.path.splitext(os.path.basename(SOURCE_FILE))[0]
    output_file = os.path.join(EXTRACTED_DATA_OUTPUT_DIR, f"{base_name}_extraction.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([doc], f, indent=4, ensure_ascii=False)

    print(f"\nSaved extraction to {output_file}")
