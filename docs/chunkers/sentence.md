# chunkers/sentence.py

Sentence-level chunking. Groups a fixed number of consecutive sentences
into each chunk using a real sentence tokenizer.

## Logic

1. Reconstruct the full text and metadata map from atomic units.
2. Split the text with nltk.tokenize.sent_tokenize. The Punkt tokenizer
   is downloaded on first import if it is not already present locally.
   Unlike the naive regex (?<=[.!?])\\s+, Punkt handles common
   abbreviations, decimals and common edge cases correctly.
3. Walk the sentence list in steps of sentences_per_chunk and join each
   window with a single space to form one chunk.
4. Resolve provenance with find_metadata_for_chunk.
5. Return chunk dicts with chunk_id sentence_i where i is the window
   index.

## Why Punkt

A regex splitter mis-splits phrases like e.g., Mr., 1.5%, www.site.com
and penalizes the sentence strategy on any realistic document. nltk is
the de-facto standard for sentence segmentation in IR benchmarks and
it keeps the benchmark comparison honest.
