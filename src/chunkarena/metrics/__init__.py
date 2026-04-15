"""Metrics package.

Re-exports every individual metric so evaluation.runner can import them from
a single place while each formula stays in its own file.
"""

from .embedding import get_embedding, embedder, embedding_cache, relevance_cache
from .relevance import is_relevant
from .composite import threshold_verdict

from .retrieval.hit import hit_at_k
from .retrieval.mrr import mrr_score
from .retrieval.precision import precision_at_k
from .retrieval.ndcg import ndcg_at_k
from .retrieval.recall import recall_at_k
from .retrieval.avg_rank import avg_rank_score

from .structural.redundancy import redundancy_score
from .structural.boundary import boundary_score
from .structural.token_cost import token_cost

from .quality.rag_quality import context_relevance, faithfulness, answer_correctness

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
