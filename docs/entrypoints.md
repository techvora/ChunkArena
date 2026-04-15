# Entrypoints: chunking.py and evaluate.py

Both top-level scripts are intentionally thin. They exist so you can
run the pipeline with python chunking.py and python evaluate.py
without remembering the package paths.

## chunking.py

- Reads config.NORMALIZED_FILE.
- Ensures config.CHUNK_OUTPUT_DIR exists.
- Iterates config.CHUNK_METHODS. For each method:
    - Dispatches chunk_normalized_documents with the method-specific
      kwargs (chunk_size and overlap for size-based strategies,
      sentences_per_chunk for sentence, nothing for paragraph, header
      and semantic).
    - Writes the result to chunks_<method>.json under the output
      directory.
- No chunking logic lives in this file.

## evaluate.py

- Imports run from the evaluation package and calls it under the
  __main__ guard.
- No evaluation logic lives in this file. Everything is in
  evaluation/runner.py and its supporting modules.

## Why they look empty

The rest of the code was deliberately pushed into packages so each
strategy and each metric can be read, reviewed and modified in
isolation without scrolling through unrelated code.
