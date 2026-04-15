# chunkers/paragraph.py

Paragraph-level chunking. The simplest strategy in the benchmark: trust
the normalizer's atomic unit types and emit one chunk per unit of type
paragraph.

## Logic

1. Iterate the raw atomic units without reconstructing the text.
2. Skip any unit whose type is not paragraph (headings, images, tables,
   formulas are dropped from the chunk set).
3. For each paragraph, emit a chunk dict with chunk_id para_i, the raw
   content as text, and a metadata block containing the single unit id,
   the type paragraph, and the position.

## Caveats

Because non-paragraph units are skipped, any answer that lives entirely
inside a table, formula, or list under a heading will be unreachable by
this strategy. That is the honest behavior of paragraph chunking and the
benchmark will reflect it through lower recall.
