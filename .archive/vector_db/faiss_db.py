# vector_db_faiss.py
import faiss
import numpy as np
import json

class FAISSVectorDB:
    def __init__(self, dimension=384):  # all-MiniLM-L6-v2 = 384 dims
        self.dimension = dimension
        self.index = None
        self.chunks = []  # store chunk metadata

    def create_index(self, chunks, index_name=None):
        # chunks: list of dict with "text", "chunk_id"
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(self.dimension)  # inner product = cosine if normalized
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        print(f"FAISS index created with {self.index.ntotal} vectors")

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "text": self.chunks[idx]["text"],
                    "score": float(scores[0][i])
                })
        return results