# chunkers/base.py

Shared infrastructure for every chunking strategy.

## Responsibilities

- Load BGE-M3 once through langchain_huggingface.HuggingFaceEmbeddings.
  The device is cuda if available, else cpu. The model is loaded eagerly
  at import time because SemanticChunker needs it during instantiation.
- Build semantic_splitter (langchain_experimental SemanticChunker with the
  percentile breakpoint strategy). Only semantic.py uses it but it is
  constructed here so the heavy model loads exactly once.
- reconstruct_text_and_metadata(units)
    Rebuilds a single markdown-like text string from the list of
    normalized atomic units. Headings get the correct number of hash
    prefixes, paragraphs go through as-is, images and tables and
    formulas become bracketed placeholders, and two newlines separate
    every unit for clean downstream splitting. Alongside the text it
    returns a metadata_map: a list of per-unit records with start and
    end character offsets, unit_id, unit_type, position and heading
    level.
- find_metadata_for_chunk(chunk_text, full_text, metadata_map)
    Given an emitted chunk substring, finds its first occurrence in the
    reconstructed text, computes its end offset, and returns the list of
    unit ids whose byte ranges overlap it. When the chunk cannot be
    located in the reconstructed text (because the splitter normalized
    whitespace or tokens), the function returns an empty unit_ids list
    with char_start and char_end set to -1 and the explicit flag
    metadata_recovery_failed = True, so the failure is visible instead
    of being silently attributed to unit 0.

## How it is used

Every strategy module imports reconstruct_text_and_metadata and
find_metadata_for_chunk. semantic.py also imports semantic_splitter.
paragraph.py does not call either because it operates directly on
typed atomic units.
