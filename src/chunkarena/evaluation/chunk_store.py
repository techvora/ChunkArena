"""Chunk store and BM25 index builder.

Scrolls every Qdrant collection to read the full set of stored chunks, keeps
them in module-level caches keyed by method name, and builds a BM25Okapi
index per method so hybrid retrieval can run without touching Qdrant for the
lexical half.

Exposed state:
    all_chunks_cache[method]     = (ids, texts)
    all_chunks_text_dict[method] = {id: text}
    bm25_models[method]          = BM25Okapi
"""

import re

from rank_bm25 import BM25Okapi

from chunkarena.config import CHUNK_METHODS, COLLECTION_NAMES
from .models import client


word_re = re.compile(r"\w+")

all_chunks_cache = {}
all_chunks_text_dict = {}
bm25_models = {}


def get_all_chunks(collection_name: str):
    """Scroll a Qdrant collection and return all stored chunk ids and texts.

    Pages through the collection in batches of 1000 points with payloads
    and accumulates the full set into memory. Used at evaluation start-up
    to build the BM25 index and the text lookup cache; callers assume the
    whole collection fits in RAM, which is fine for the shipped banking
    corpus and for any collection up to a few hundred thousand chunks.

    Args:
        collection_name: Qdrant collection name (usually a chunking
            method identifier such as ``"fixed_size"`` or ``"semantic"``).

    Returns:
        Tuple ``(ids, texts)`` of parallel lists ordered by Qdrant's
        scroll order.
    """
    all_points, offset = [], None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    ids   = [p.id for p in all_points]
    texts = [p.payload["text"] for p in all_points]
    return ids, texts


def build_all_stores():
    """Populate the module-level caches for every chunking method.

    For each method listed in :data:`config.CHUNK_METHODS`, scrolls the
    corresponding Qdrant collection, stores ``(ids, texts)`` in
    :data:`all_chunks_cache`, builds an id-to-text lookup in
    :data:`all_chunks_text_dict`, and fits a ``BM25Okapi`` index in
    :data:`bm25_models` over whitespace-lowercased word tokens. Must be
    called once before any retrieval helper runs; the caches are
    process-global so the evaluation loop can reuse them across every
    (method, technique, question) triple.
    """
    print("Loading collections and building BM25 indexes...")
    for method in CHUNK_METHODS:
        coll = COLLECTION_NAMES.get(method, method)
        ids, texts = get_all_chunks(coll)
        all_chunks_cache[method]     = (ids, texts)
        all_chunks_text_dict[method] = dict(zip(ids, texts))
        tokenized = [word_re.findall(t.lower()) for t in texts]
        bm25_models[method] = BM25Okapi(tokenized)
        print(f"  {method}: {len(texts)} chunks")
