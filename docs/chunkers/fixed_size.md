# chunkers/fixed_size.py

Fixed-size chunking using LangChain's RecursiveCharacterTextSplitter.

## Logic

1. Reconstruct the full markdown text and metadata map from the atomic
   units via base.reconstruct_text_and_metadata.
2. Instantiate RecursiveCharacterTextSplitter with chunk_size, chunk_overlap
   and the separator ladder ["\\n\\n", "\\n", " ", ""]. The ladder means the
   splitter first tries to cut on paragraph breaks, then newlines, then
   spaces, and only resorts to character cuts when nothing better fits the
   size budget.
3. Split the text and, for each produced chunk, resolve its provenance via
   base.find_metadata_for_chunk.
4. Return a list of dicts with chunk_id, text and metadata.

## Reuse note

overlapping and recursive strategies delegate to fixed_size_chunk directly
through the registry in chunkers/__init__.py. They are separate benchmark
entries only; there is no dedicated module for them because the only
difference is the overlap parameter passed by chunking.py.
