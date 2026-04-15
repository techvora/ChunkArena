# chunkers/header.py

Header-aware chunking using LangChain's MarkdownHeaderTextSplitter.

## Logic

1. Reconstruct the markdown text and metadata map. Because base.py writes
   real markdown heading prefixes (#, ##, ###) when it encounters a
   heading unit, the splitter can navigate the document structurally.
2. Configure the splitter to split on H1, H2 and H3 levels.
3. Call split_text, which returns a list of Document objects. Each
   document has a page_content holding the section body and a metadata
   dict holding the nearest header at each configured level.
4. For each split, resolve provenance via find_metadata_for_chunk and
   merge in the header dict under chunk_meta["headers"].
5. Return the list of chunk dicts with chunk_id header_i.

## Why it is useful

Sections are the natural retrieval unit for structured documents like
banking policies. A hit on the right section tends to include all the
context needed to answer a question, which raises recall at the cost of
longer chunks.
