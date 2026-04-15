# chunkers/recursive.py

Recursive character chunking with a rich separator hierarchy. This is
the canonical recursive chunking configuration reported in benchmark
literature.

## Logic

1. Reconstruct the full markdown text and metadata map.
2. Instantiate RecursiveCharacterTextSplitter with the separator ladder
   ["\\n\\n", "\\n", ". ", "! ", "? ", "; ", ", ", " ", ""]. The
   splitter walks the ladder top-down and tries to cut at the earliest
   level that fits the chunk_size budget.
3. Split the text and resolve provenance for each chunk via
   find_metadata_for_chunk.
4. Return chunk dicts with chunk_id recursive_i.

## Why it is distinct from fixed_size and overlapping

fixed_size and overlapping share the shorter separator ladder
["\\n\\n", "\\n", " ", ""]. Recursive walks a richer hierarchy with
sentence terminators and clause separators before falling through to
spaces, which produces genuinely different boundaries on realistic
prose. Keeping it separate from fixed_size/overlapping is the only
reason it earns its own row in the benchmark.
