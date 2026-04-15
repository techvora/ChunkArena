"""Stage 4 — Qdrant vector store adapter.

Owns everything specific to Qdrant so the rest of the framework can stay
backend-agnostic:

    * Loads the shared embedding model (BAAI/bge-m3 by default) once per
      process and uses the same instance at index-time and query-time so
      query vectors live in the same space as the stored points.
    * Creates one collection per chunking method (fixed_size, semantic,
      recursive, ...) and indexes the corresponding chunks_<method>.json
      produced by Stage 3.
    * Exposes both dense and hybrid (dense + BM25) retrieval used by the
      evaluation runner for the various retrieval techniques.

Every tunable (host, port, embedding model, embedding dimension, device)
is read from chunkarena.config so swapping Qdrant for a different store
is a matter of writing a parallel module that implements the same public
methods — no config edits needed elsewhere.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from typing import List, Dict, Any
from chunkarena.config import (
    CHUNKS_PATH,
    DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    QDRANT_HOST,
    QDRANT_PORT,
)

class QdrantVectorDB:
    """Qdrant-backed vector store adapter for the ChunkArena pipeline.

    Encapsulates the shared embedding model (loaded once per instance),
    the Qdrant client connection and the naming / versioning conventions
    used by the indexing stage. Each chunking method gets its own
    collection; callers never touch the underlying client directly. An
    alternative backend (Pinecone, Weaviate, pgvector) can be dropped in
    by implementing the same public surface: ``create_index``,
    ``search``, ``list_collections``, ``delete_collection``.
    """

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        """Connect to Qdrant and load the shared embedding model.

        The SentenceTransformer is held on the instance so every call to
        :meth:`create_index` and :meth:`search` embeds its inputs with
        the identical model, guaranteeing that query and document
        vectors live in the same space.

        Args:
            host: Qdrant host; defaults to :data:`config.QDRANT_HOST`.
            port: Qdrant REST port; defaults to :data:`config.QDRANT_PORT`.
        """
        self.client = QdrantClient(host=host, port=port)
        self.dimension = EMBEDDING_DIMENSION
        self.device = DEVICE
        print(f"Loading {EMBEDDING_MODEL} on {self.device}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

    def create_index(
        self,
        chunks: List[Dict],
        index_name: str,
        if_exists: str = "version",
    ) -> str:
        """Create or version a Qdrant collection and index a chunk set.

        Sanitizes ``index_name``, then applies the requested collision
        policy if the collection already exists. Builds the collection
        with cosine distance at :data:`config.EMBEDDING_DIMENSION`,
        encodes every chunk with the shared SentenceTransformer (batch
        size 12, normalized vectors) and upserts the points in batches
        of 256 with a payload carrying ``chunk_id``, ``text``,
        ``metadata`` and optional ``source_doc``.

        Args:
            chunks: Chunk dicts with ``chunk_id``, ``text`` and
                ``metadata`` keys.
            index_name: Human-readable collection name; usually a
                chunking-method identifier.
            if_exists: Collision policy when the base collection already
                exists. One of ``"version"`` (create ``<name>_v2``,
                ``_v3``, ... — the default), ``"overwrite"`` (drop and
                recreate), ``"skip"`` (leave existing collection alone
                and return its name) or ``"error"``.

        Returns:
            The actual collection name that points were written to.

        Raises:
            ValueError: If ``if_exists`` is ``"error"`` and the
                collection exists, or if ``if_exists`` is not one of the
                supported modes.
        """
        base_name = self._sanitize_name(index_name)
        collection_name = base_name

        if self.client.collection_exists(base_name):
            if if_exists == "overwrite":
                self.client.delete_collection(collection_name=base_name)
            elif if_exists == "skip":
                print(f"Collection '{base_name}' already exists, skipping.")
                return base_name
            elif if_exists == "error":
                raise ValueError(f"Collection '{base_name}' already exists.")
            elif if_exists == "version":
                collection_name = self._next_version_name(base_name)
                print(f"Collection '{base_name}' exists — writing new version '{collection_name}'.")
            else:
                raise ValueError(f"Unknown if_exists mode: {if_exists!r}")

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
        )

        texts = [c["text"] for c in chunks]

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=12,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Build payload from chunk metadata
            payload = {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})  # includes unit_ids, positions, etc.
            }
            # Also add any top-level fields you might need for filtering
            if "source_doc" in chunk:
                payload["source_doc"] = chunk["source_doc"]
            points.append(PointStruct(
                id=i,
                vector=emb.tolist(),
                payload=payload
            ))

        batch_size = 256
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[start:start + batch_size],
            )
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"Collection '{collection_name}' populated with {len(points)} points.")
        return collection_name

    def _next_version_name(self, base_name: str) -> str:
        """Return the next unused ``<base_name>_vN`` suffix for versioning.

        Lists existing collections and picks the smallest ``N >= 2`` for
        which ``<base_name>_vN`` is free. Used by :meth:`create_index`
        when ``if_exists="version"``.

        Args:
            base_name: Already-sanitized base collection name.

        Returns:
            A collection name of the form ``<base_name>_vN`` that is
            guaranteed not to collide with an existing collection at the
            time of the call.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        n = 2
        while f"{base_name}_v{n}" in existing:
            n += 1
        return f"{base_name}_v{n}"

    def search(self, query_text: str, index_name: str, top_k: int = 5) -> List[Dict]:
        """Dense cosine search against a single chunking-method collection.

        Embeds ``query_text`` with the same SentenceTransformer used at
        index time (no separate query encoder), then asks Qdrant for the
        top-``top_k`` nearest points with payloads attached.

        Args:
            query_text: Natural-language query.
            index_name: Collection name; sanitized the same way as in
                :meth:`create_index`.
            top_k: Number of nearest neighbours to return.

        Returns:
            List of dicts with keys ``chunk_id``, ``text``, ``score``
            (cosine similarity) and ``metadata``.

        Raises:
            ValueError: If the target collection does not exist.
        """
        collection_name = self._sanitize_name(index_name)
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist. Create it first.")

        # Use the same SentenceTransformer instance used at index time so query
        # and document embeddings live in an identical vector space.
        with torch.no_grad():
            query_embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id"),
                "text": hit.payload.get("text"),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in response.points
        ]

    def list_collections(self) -> List[str]:
        """List every collection currently present in the Qdrant instance.

        Returns:
            List of raw collection names. Includes any version suffixes
            created by :meth:`create_index`.
        """
        return [c.name for c in self.client.get_collections().collections]

    def delete_collection(self, index_name: str):
        """Drop a single collection, sanitizing the name first.

        Args:
            index_name: Logical collection name to delete; passed through
                :meth:`_sanitize_name` so callers may use the same casing
                they used at create time.
        """
        collection_name = self._sanitize_name(index_name)
        self.client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted.")

    def _sanitize_name(self, name: str) -> str:
        """Normalize a user-supplied name into a valid collection name.

        Lowercases and replaces spaces with underscores. Centralised so
        every public method applies the same transformation and callers
        do not need to remember collection-name rules.

        Args:
            name: Raw name as supplied by the caller.

        Returns:
            Sanitized collection name.
        """
        return name.replace(" ", "_").lower()


# ------------------------------------------------------------
# Example usage: index all chunking methods
# ------------------------------------------------------------
if __name__ == "__main__":
    # Directory where chunk JSON files are stored (from your chunking script)
    CHUNKS_DIR = CHUNKS_PATH
    # List of methods (must match filenames: chunks_{method}.json)
    METHODS = ["fixed_size", "overlapping", "sentence", "paragraph", "recursive", "header", "semantic"]

    qdrant = QdrantVectorDB()

    for method in METHODS:
        file_path = os.path.join(CHUNKS_DIR, f"chunks_{method}.json")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {method}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Indexing {method} with {len(chunks)} chunks...")
        qdrant.create_index(chunks, index_name=method)

    print("\nAll collections created. Available collections:")
    print(qdrant.list_collections())

    # Example search on 'fixed_size' collection
    # query = "What is fractional reserve banking?"
    # print(f"\nSearching 'fixed_size' for: {query}")
    # results = qdrant.search(query, index_name="fixed_size", top_k=3)
    # for r in results:
    #     print(f"  - {r['chunk_id']} (score={r['score']:.4f}): {r['text'][:100]}...")