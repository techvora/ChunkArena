"""Metrics package.

Re-exports every individual metric so evaluation.runner can import them from
a single place while each formula stays in its own file.
"""

from .embedding import get_embedding, embedder, embedding_cache, relevance_cache
from .relevance import is_relevant
from .hit import hit_at_k
from .mrr import mrr_score
from .precision import precision_at_k
from .ndcg import ndcg_at_k
from .recall import recall_at_k
from .avg_rank import avg_rank_score
from .redundancy import redundancy_score
from .boundary import boundary_score
from .token_cost import token_cost
from .rag_quality import context_relevance, faithfulness, answer_correctness
from .composite import threshold_verdict

__all__ = [
    "get_embedding",
    "embedder",
    "embedding_cache",
    "relevance_cache",
    "is_relevant",
    "hit_at_k",
    "mrr_score",
    "precision_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "avg_rank_score",
    "redundancy_score",
    "boundary_score",
    "token_cost",
    "context_relevance",
    "faithfulness",
    "answer_correctness",
    "threshold_verdict",
]
