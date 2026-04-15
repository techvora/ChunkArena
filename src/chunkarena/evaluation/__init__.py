"""Evaluation package.

Exposes the runner entrypoint. Sub-modules (models, data_loader, chunk_store,
chunk_stats, retrieval, report_excel) are imported lazily by the runner so
importing this package does not load heavy models unless the runner is used.
"""

from .runner import run

__all__ = ["run"]
