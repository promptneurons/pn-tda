"""Tests for NDCG evaluation and experiment harness."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from examples.pn_autoresearch_integration.evaluate import (
    evaluate_search_quality,
    mrr,
    ndcg_at_k,
    precision_at_k,
)


# --- Metric unit tests ---


def test_ndcg_perfect_ranking():
    """All relevant docs at top → NDCG = 1.0."""
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(ranked, relevant, k=5) == 1.0


def test_ndcg_worst_ranking():
    """Relevant docs at bottom → NDCG < 1.0."""
    ranked = ["x", "y", "z", "a", "b"]
    relevant = {"a", "b"}
    score = ndcg_at_k(ranked, relevant, k=5)
    assert 0.0 < score < 1.0


def test_ndcg_no_relevant():
    """No relevant docs in ranking → NDCG = 0.0."""
    ranked = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_empty_relevant():
    """Empty relevant set → NDCG = 0.0."""
    ranked = ["a", "b"]
    assert ndcg_at_k(ranked, set(), k=5) == 0.0


def test_mrr_first_relevant():
    """First result relevant → MRR = 1.0."""
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_second_relevant():
    """Second result relevant → MRR = 0.5."""
    assert mrr(["x", "a", "b"], {"a"}) == 0.5


def test_mrr_none_relevant():
    """No relevant results → MRR = 0.0."""
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_precision_at_3_all():
    """All top-3 relevant → P@3 = 1.0."""
    assert precision_at_k(["a", "b", "c", "d"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_3_none():
    """No top-3 relevant → P@3 = 0.0."""
    assert precision_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_precision_at_3_partial():
    """One of 3 relevant → P@3 = 1/3."""
    result = precision_at_k(["a", "x", "y"], {"a"}, k=3)
    assert abs(result - 1 / 3) < 1e-9


# --- Evaluate integration ---


def test_evaluate_returns_all_metrics():
    queries = [
        {"query": "test", "relevant": ["a", "b"]},
        {"query": "other", "relevant": ["c"]},
    ]

    def search(q):
        return ["a", "b", "c"]

    metrics = evaluate_search_quality(search, queries)
    assert "ndcg_at_10" in metrics
    assert "mrr" in metrics
    assert "precision_at_3" in metrics
    assert metrics["num_queries"] == 2


def test_evaluate_empty_queries():
    metrics = evaluate_search_quality(lambda q: [], [])
    assert metrics["num_queries"] == 0
    assert metrics["ndcg_at_10"] == 0.0


# --- Experiment integration ---


def test_experiment_runs():
    """The full experiment script should run without errors."""
    # Import and run the experiment's create functions + search
    from examples.pn_autoresearch_integration.experiment import (
        create_corpus_db,
        create_signal_db,
        keyword_jaccard,
        make_baseline_search,
    )
    from examples.pn_autoresearch_integration.evaluate import load_queries

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path = str(Path(tmpdir) / "signals.db")
        corpus_path = str(Path(tmpdir) / "corpus.db")
        create_signal_db(signal_path)
        create_corpus_db(corpus_path)

        conn = sqlite3.connect(corpus_path)
        conn.row_factory = sqlite3.Row
        baseline = make_baseline_search(conn)

        queries_path = str(
            Path(__file__).parent.parent
            / "examples/pn_autoresearch_integration/queries.jsonl"
        )
        queries = load_queries(queries_path)
        metrics = evaluate_search_quality(baseline, queries)
        conn.close()

        assert metrics["num_queries"] == 10
        assert 0.0 <= metrics["ndcg_at_10"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0


def test_tda_differs_from_baseline():
    """TDA experiment should produce different scores than baseline."""
    from examples.pn_autoresearch_integration.experiment import (
        create_corpus_db,
        create_signal_db,
        make_baseline_search,
        make_tda_search,
    )
    from examples.pn_autoresearch_integration.evaluate import load_queries
    from examples.pn_autoresearch_integration.tda_scorer import TDAScorer

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path = str(Path(tmpdir) / "signals.db")
        corpus_path = str(Path(tmpdir) / "corpus.db")
        create_signal_db(signal_path)
        create_corpus_db(corpus_path)

        conn = sqlite3.connect(corpus_path)
        conn.row_factory = sqlite3.Row

        baseline = make_baseline_search(conn)

        scorer = TDAScorer()
        scorer.configure({"signal_db": signal_path, "epsilon_max": 1.0})
        scorer.precompute(conn)
        tda = make_tda_search(conn, scorer, tda_weight=0.3)

        queries_path = str(
            Path(__file__).parent.parent
            / "examples/pn_autoresearch_integration/queries.jsonl"
        )
        queries = load_queries(queries_path)

        baseline_metrics = evaluate_search_quality(baseline, queries)
        tda_metrics = evaluate_search_quality(tda, queries)
        conn.close()

        # TDA fusion should change at least one metric
        assert (
            baseline_metrics["ndcg_at_10"] != tda_metrics["ndcg_at_10"]
            or baseline_metrics["mrr"] != tda_metrics["mrr"]
            or baseline_metrics["precision_at_3"] != tda_metrics["precision_at_3"]
        ), "TDA should produce at least one different metric from baseline"
