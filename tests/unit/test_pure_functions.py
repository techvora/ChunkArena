"""Unit tests for pure helpers.

Scoped deliberately to functions that do NOT load the embedding model,
reranker, or touch Qdrant — so the suite runs in under a second and has
zero network / GPU / disk dependencies. Integration tests that exercise
the full pipeline belong under tests/integration/.
"""

import math

import pytest

from chunkarena.config import (
    COMPOSITE_WEIGHTS,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from chunkarena.metrics.composite import threshold_verdict
from chunkarena.normalization.normalizer import clean_text, clean_url, is_heading


# ------------------------------------------------------------------
# 1. normalizer.clean_text — whitespace collapsing + escape unwrap
# ------------------------------------------------------------------
class TestCleanText:
    def test_collapses_runs_of_whitespace_to_single_space(self):
        assert clean_text("hello   \n\t  world") == "hello world"

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("   padded text   ") == "padded text"

    def test_unwraps_pdf_escaped_underscore(self):
        # docling emits "foo\_bar" for foo_bar in some PDFs — clean_text must
        # restore the literal underscore or downstream metadata joins break.
        assert clean_text("foo\\_bar baz") == "foo_bar baz"

    def test_empty_string_is_preserved(self):
        assert clean_text("") == ""

    def test_whitespace_only_becomes_empty(self):
        assert clean_text("   \n\t  ") == ""


# ------------------------------------------------------------------
# 2. normalizer.clean_url — strips spaces that PDFs inject into URLs
# ------------------------------------------------------------------
class TestCleanUrl:
    def test_removes_spaces_inside_url(self):
        assert clean_url("https://exa mple.com/pa th") == "https://example.com/path"

    def test_leaves_clean_url_untouched(self):
        assert clean_url("https://example.com/path?q=1") == "https://example.com/path?q=1"

    def test_removes_newlines_and_tabs(self):
        assert clean_url("https://example.com/\n\tfoo") == "https://example.com/foo"


# ------------------------------------------------------------------
# 3. normalizer.is_heading — markdown heading detection
# ------------------------------------------------------------------
class TestIsHeading:
    def test_detects_h1(self):
        assert is_heading("# Title") == (True, 1)

    def test_detects_h3(self):
        assert is_heading("### Section") == (True, 3)

    def test_caps_level_at_six(self):
        # Markdown only supports h1..h6, so `####### foo` must clamp to 6.
        assert is_heading("####### too deep") == (True, 6)

    def test_rejects_plain_text(self):
        assert is_heading("just a paragraph") == (False, 0)

    def test_rejects_hash_without_space(self):
        # "#notaheading" is a hashtag, not a markdown heading.
        assert is_heading("#notaheading") == (False, 0)

    def test_image_line_is_not_a_heading(self):
        # Image lines are rendered as `# Image:` by the extractor but must be
        # classified as content, not structural headings.
        assert is_heading("# Image:") == (False, 0)

    def test_leading_whitespace_tolerated(self):
        assert is_heading("   ## Indented") == (True, 2)


# ------------------------------------------------------------------
# 4. composite.threshold_verdict — verdict banding incl. redundancy flip
# ------------------------------------------------------------------
class TestThresholdVerdict:
    def test_higher_is_better_good_band(self):
        assert threshold_verdict("ndcg_at_k", 0.9) == "Good"

    def test_higher_is_better_moderate_band(self):
        assert threshold_verdict("ndcg_at_k", 0.6) == "Moderate"

    def test_higher_is_better_bad_band(self):
        assert threshold_verdict("ndcg_at_k", 0.2) == "Bad"

    def test_redundancy_direction_is_inverted(self):
        # For redundancy, LOWER is better — a value of 0.1 must be Good, not Bad.
        assert threshold_verdict("redundancy", 0.1) == "Good"
        assert threshold_verdict("redundancy", 0.9) == "Bad"

    def test_unknown_metric_returns_na(self):
        assert threshold_verdict("nonexistent_metric", 0.5) == "N/A"

    def test_nan_value_returns_na(self):
        assert threshold_verdict("ndcg_at_k", float("nan")) == "N/A"


# ------------------------------------------------------------------
# 5. config — central infra constants exist and are self-consistent
# ------------------------------------------------------------------
class TestConfigInvariants:
    def test_composite_weights_sum_to_one(self):
        # config.py asserts this at import time, but we re-verify here so a
        # regression surfaces as a clean test failure instead of an ImportError.
        total = sum(COMPOSITE_WEIGHTS.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6), (
            f"COMPOSITE_WEIGHTS must sum to 1.0, got {total}"
        )

    def test_all_main_ir_metrics_are_weighted(self):
        expected = {"ndcg_at_k", "mrr", "hit_at_k", "recall_at_k", "precision_at_k"}
        assert set(COMPOSITE_WEIGHTS.keys()) == expected

    def test_central_models_infra_constants_are_wired(self):
        # These were hardcoded in 4 different files before centralization.
        # If any regression reintroduces a hardcoded "BAAI/bge-m3" or
        # "localhost" string, the config surface is the canonical source.
        assert isinstance(EMBEDDING_MODEL, str) and EMBEDDING_MODEL
        assert EMBEDDING_DIMENSION > 0
        assert isinstance(QDRANT_HOST, str) and QDRANT_HOST
        assert 1 <= QDRANT_PORT <= 65535
