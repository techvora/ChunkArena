"""Retrieval and reranking model instances.

Loads the cross-encoder used for reranking and the Qdrant client used for
dense and hybrid search. Kept separate from metrics/embedding.py so that
metric code never has to know about Qdrant, and retrieval code never has to
import the embedder directly.
"""

from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient


device = "cpu"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
client = QdrantClient(host="localhost", port=6333)
