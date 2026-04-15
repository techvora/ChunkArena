"""Retrieval and reranking model instances.

Loads the cross-encoder used for reranking and the Qdrant client used for
dense and hybrid search. Kept separate from metrics/embedding.py so that
metric code never has to know about Qdrant, and retrieval code never has to
import the embedder directly.
"""

from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient

from chunkarena.config import DEVICE, RERANKER_MODEL, QDRANT_HOST, QDRANT_PORT


device = DEVICE
cross_encoder = CrossEncoder(RERANKER_MODEL, device=device)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
