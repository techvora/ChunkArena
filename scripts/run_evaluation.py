"""Evaluation entrypoint.

Thin wrapper that delegates to chunkarena.evaluation.runner.run. All metric,
retrieval and reporting logic lives under the chunkarena.metrics and
chunkarena.evaluation packages.
"""

from chunkarena.evaluation import run


if __name__ == "__main__":
    run()
