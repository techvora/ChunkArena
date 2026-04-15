"""Extraction entrypoint.

Thin wrapper around chunkarena.extraction.pipeline. Extracts text from every
document under data/raw and writes structured JSON to data/bronze/extracted.
"""

from chunkarena.extraction.pipeline import process_complex_folder
from chunkarena.config import SOURCE_FILE, EXTRACTED_DATA_OUTPUT_DIR
import os


if __name__ == "__main__":
    raw_dir = os.path.dirname(SOURCE_FILE)
    os.makedirs(EXTRACTED_DATA_OUTPUT_DIR, exist_ok=True)
    process_complex_folder(raw_dir)
