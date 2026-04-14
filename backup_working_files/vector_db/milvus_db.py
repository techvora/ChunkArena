# vector_db_milvus.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np

class MilvusVectorDB:
    def __init__(self, host="localhost", port="19530", dimension=384):
        connections.connect(host=host, port=port)
        self.dimension = dimension
        self.collection_name = None

    def create_index(self, chunks, index_name):
        self.collection_name = index_name.replace(" ", "_")
        # Drop if exists
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=200)
        ]
        schema = CollectionSchema(fields, description="Chunks")
        col = Collection(self.collection_name, schema)
        # Embed
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        entities = [
            embeddings.tolist(),
            [c["chunk_id"] for c in chunks],
            [c["text"] for c in chunks],
            [c["source_doc"] for c in chunks]
        ]
        col.insert(entities)
        # Create index
        index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
        col.create_index(field_name="vector", index_params=index_params)
        col.load()
        print(f"Milvus collection {self.collection_name} ready with {col.num_entities} entities")

    def search(self, query_embedding, top_k=5):
        col = Collection(self.collection_name)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = col.search(
            data=[query_embedding.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "text", "source_doc"]
        )
        hits = results[0]
        return [{"chunk_id": hit.entity.get("chunk_id"), "text": hit.entity.get("text"), "score": hit.score} for hit in hits]