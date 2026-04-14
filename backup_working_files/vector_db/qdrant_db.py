from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch

class QdrantVectorDB:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        # BGE-M3 default dense dimension is 1024
        self.dimension = 1024 
        self.collection_name = None
        
        # Determine device and load model once
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BGE-M3 on {self.device}...")
        
        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        
        # Optimization for 4GB GPU: Use half precision (FP16)
        if self.device == "cuda":
            self.model.half() 

    def create_index(self, chunks, index_name):
        self.collection_name = index_name.replace(" ", "_").lower()
        
        # Recreate collection with 1024 dimensions
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
        )
        
        # Prepare texts
        texts = [c["text"] for c in chunks]
        
        # Generate embeddings
        # Batch size 8-16 is safe for 4GB VRAM with BGE-M3
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, 
                batch_size=12, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
        
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={
                    "chunk_id": chunk.get("chunk_id"), 
                    "text": chunk["text"], 
                    "source_doc": chunk["source_doc"]
                }
            ))
            
        self.client.upsert(collection_name=self.collection_name, points=points)
        
        # Clear VRAM cache after heavy indexing
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        print(f"Qdrant collection {self.collection_name} populated with {len(points)} points")

    def search(self, query_text, top_k=5):
        """
        Modified to take 'query_text' directly to handle embedding internally.
        """
        # Embed the text query using the same model
        with torch.no_grad():
            query_embedding = self.model.encode(query_text, convert_to_numpy=True)
            
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return [
            {
                "chunk_id": hit.payload.get("chunk_id"), 
                "text": hit.payload.get("text"), 
                "score": hit.score
            } for hit in results
        ]