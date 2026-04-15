# metrics/__init__.py

Flat re-export of every metric and the shared embedding helper. The
evaluation runner imports everything it needs from metrics directly so
it never has to know the individual module names:

    from metrics import (
        get_embedding, hit_at_k, mrr_score, precision_at_k, ndcg_at_k,
        recall_at_k, avg_rank_score, redundancy_score, boundary_score,
        token_cost, threshold_verdict,
    )

Adding a new metric is a three-step operation: write the module under
metrics/, import and re-export it from metrics/__init__.py, and plug
it into evaluation.runner's main loop and summary aggregation.
