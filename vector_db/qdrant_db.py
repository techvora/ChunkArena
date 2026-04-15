from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from typing import List, Dict, Any
from config import CHUNKS_PATH

class QdrantVectorDB:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.dimension = 1024  # BGE-M3 dense dimension
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print(f"Loading BGE-M3 on {self.device}...")
        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        # if self.device == "cuda":
        #     self.model.half()  # FP16 for 4GB GPU

    def create_index(
        self,
        chunks: List[Dict],
        index_name: str,
        if_exists: str = "version",
    ) -> str:
        """
        Create a Qdrant collection for a specific chunking method.
        :param chunks: list of chunk dicts (with 'chunk_id', 'text', 'metadata')
        :param index_name: e.g., "fixed_size", "semantic", "header"
        :param if_exists: behavior when a collection with this name already exists:
            - "version":   create a new collection with suffix _v2, _v3, ... (default)
            - "overwrite": delete the existing collection and recreate it
            - "skip":      do nothing, return the existing collection name
            - "error":     raise ValueError
        :return: the actual collection name that was written to.
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
        """Return base_name_vN where N is the smallest integer >= 2 not in use."""
        existing = {c.name for c in self.client.get_collections().collections}
        n = 2
        while f"{base_name}_v{n}" in existing:
            n += 1
        return f"{base_name}_v{n}"

    def search(self, query_text: str, index_name: str, top_k: int = 5) -> List[Dict]:
        """
        Search a specific collection (chunking method).
        :param query_text: user query string
        :param index_name: name of the collection (e.g., "fixed_size")
        :param top_k: number of results to return
        :return: list of dicts with chunk_id, text, score, and metadata
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
        """Return list of all collection names (chunking methods) stored."""
        return [c.name for c in self.client.get_collections().collections]

    def delete_collection(self, index_name: str):
        """Delete a specific collection (e.g., to re-index)."""
        collection_name = self._sanitize_name(index_name)
        self.client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted.")

    def _sanitize_name(self, name: str) -> str:
        """Convert any string to a valid Qdrant collection name."""
        return name.replace(" ", "_").lower()


# ------------------------------------------------------------
# Example usage: index all chunking methods
# ------------------------------------------------------------
if __name__ == "__main__":
    # Directory where chunk JSON files are stored (from your chunking script)
    CHUNKS_DIR = CHUNKS_PATH
    # List of methods (must match filenames: chunks_{method}.json)
    METHODS = ["fixed_size", "overlapping", "sentence", "paragraph", "recursive", "header", "semantic"]

    qdrant = QdrantVectorDB(host="localhost", port=6333)

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