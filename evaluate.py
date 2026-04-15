"""Evaluation entrypoint.

Thin wrapper that delegates to evaluation.runner.run. All metric, retrieval
and reporting logic lives under the metrics/ and evaluation/ packages.
"""

from evaluation import run


if __name__ == "__main__":
    run()
