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

from config import CHUNK_METHODS
from .models import client


word_re = re.compile(r"\w+")

all_chunks_cache = {}
all_chunks_text_dict = {}
bm25_models = {}


def get_all_chunks(collection_name: str):
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
    print("Loading collections and building BM25 indexes...")
    for method in CHUNK_METHODS:
        ids, texts = get_all_chunks(method)
        all_chunks_cache[method]     = (ids, texts)
        all_chunks_text_dict[method] = dict(zip(ids, texts))
        tokenized = [word_re.findall(t.lower()) for t in texts]
        bm25_models[method] = BM25Okapi(tokenized)
        print(f"  {method}: {len(texts)} chunks")
