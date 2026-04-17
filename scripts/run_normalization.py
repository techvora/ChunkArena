"""Normalization entrypoint.

Thin wrapper around chunkarena.normalization.normalizer. Reads the bronze
extraction JSON and writes a silver normalized-units JSON ready for chunking.
"""

import os
from pathlib import Path
from chunkarena.normalization.normalizer import normalize_raw_json
from chunkarena.config import EXTRACTED_DATA_OUTPUT_DIR, NORMALIZED_FILE, SOURCE_FILE


if __name__ == "__main__":
    os.makedirs(os.path.dirname(NORMALIZED_FILE), exist_ok=True)
    source_stem = Path(SOURCE_FILE).stem
    input_file = os.path.join(EXTRACTED_DATA_OUTPUT_DIR, f"{source_stem}_extraction.json")
    normalize_raw_json(input_file, NORMALIZED_FILE)
