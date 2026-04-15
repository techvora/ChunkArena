# chunkers/semantic.py

Semantic chunking via langchain_experimental.SemanticChunker. Cuts the
text at embedding-distance breakpoints so each chunk is topically
coherent rather than character-aligned.

## Logic

1. Reconstruct the full text and metadata map.
2. Call semantic_splitter.create_documents on the reconstructed text.
   The splitter (configured in base.py with breakpoint_threshold_type
   percentile) embeds each sentence with BGE-M3 and cuts where the
   semantic distance jump crosses the percentile threshold.
3. For each returned document resolve provenance via
   find_metadata_for_chunk.
4. Return chunk dicts with chunk_id semantic_i.

## Cost

This is the slowest strategy in the benchmark. Every run re-embeds the
whole document. The runtime is dominated by BGE-M3 inference rather than
any splitter logic. Worth paying when recall matters more than indexing
latency.
