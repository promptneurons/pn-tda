"""Tests for TDAScorer example integration."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from examples.pn_autoresearch_integration.tda_scorer import ScoredChunk, TDAScorer


def _create_signal_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,
            filename TEXT NOT NULL, sprint_id TEXT,
            sprint_year INTEGER, sprint_month INTEGER,
            sprint_in_month INTEGER, is_focus_hub BOOLEAN DEFAULT 0,
            title_dtg TEXT, parent_group TEXT
        );
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL, signal_type TEXT NOT NULL,
            signal_value TEXT NOT NULL, line_number INTEGER, context TEXT,
            UNIQUE(doc_id, signal_type, signal_value, line_number)
        );
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_doc_id TEXT NOT NULL, target_path TEXT NOT NULL,
            target_doc_id TEXT, edge_type TEXT NOT NULL, line_number INTEGER,
            UNIQUE(source_doc_id, target_path, edge_type, line_number)
        );

        INSERT INTO documents VALUES
            ('d1', 'a.md', 'a.md', '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d2', 'b.md', 'b.md', '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d3', 'c.md', 'c.md', '26033', 2026, 3, 3, 0, NULL, NULL);

        INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
            ('d1', 'ref', 'X', 1), ('d1', 'ref', 'Y', 2),
            ('d2', 'ref', 'X', 1), ('d2', 'ref', 'Z', 2),
            ('d3', 'ref', 'Y', 1), ('d3', 'ref', 'Z', 2);

        INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
            ('d1', 'b.md', 'd2', 'wiki_link', 1),
            ('d2', 'c.md', 'd3', 'wiki_link', 1),
            ('d3', 'a.md', 'd1', 'wiki_link', 1);
    """)
    conn.commit()
    conn.close()


def _create_corpus_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source_path TEXT NOT NULL);
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
            content TEXT NOT NULL, heading TEXT
        );

        INSERT INTO documents VALUES ('d1', 'a.md'), ('d2', 'b.md'), ('d3', 'c.md');

        INSERT INTO chunks VALUES
            ('c1', 'd1', 'First chunk of doc A.', 'Section A1'),
            ('c2', 'd1', 'Second chunk of doc A.', 'Section A2'),
            ('c3', 'd2', 'Chunk from doc B.', 'Section B1'),
            ('c4', 'd3', 'Chunk from doc C.', 'Section C1'),
            ('c5', 'd3', 'Another chunk from doc C.', 'Section C2');
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def scorer_setup(tmp_path):
    """Create DBs, configure and precompute a TDAScorer."""
    signal_path = str(tmp_path / "signals.db")
    corpus_path = str(tmp_path / "corpus.db")
    _create_signal_db(signal_path)
    _create_corpus_db(corpus_path)

    scorer = TDAScorer()
    scorer.configure({
        "signal_db": signal_path,
        "epsilon_max": 1.0,
        "max_dimension": 2,
        "num_scales": 10,
    })

    corpus_conn = sqlite3.connect(corpus_path)
    corpus_conn.row_factory = sqlite3.Row
    scorer.precompute(corpus_conn)

    yield scorer, corpus_conn
    corpus_conn.close()


def test_scorer_returns_scored_chunks(scorer_setup):
    scorer, conn = scorer_setup
    results = scorer.retrieve("test query", conn, limit=10)
    assert len(results) > 0
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_scores_in_range(scorer_setup):
    scorer, conn = scorer_setup
    results = scorer.retrieve("test query", conn, limit=10)
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range"


def test_all_chunks_scored(scorer_setup):
    """All 5 chunks should be scored (all docs are in signal DB)."""
    scorer, conn = scorer_setup
    results = scorer.retrieve("test query", conn, limit=100)
    assert len(results) == 5


def test_scores_sorted_descending(scorer_setup):
    scorer, conn = scorer_setup
    results = scorer.retrieve("test query", conn, limit=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_features_cached(scorer_setup):
    scorer, conn = scorer_setup
    features = scorer.get_features()
    assert len(features) == 3  # d1, d2, d3
    for doc_id, feat in features.items():
        assert "maturity_score" in feat
        assert "persistence_entropy_norm" in feat
        assert "betti_stability" in feat
        assert "composite" in feat


def test_no_signal_db_returns_empty(tmp_path):
    """Scorer with no signal DB should gracefully return empty."""
    corpus_path = str(tmp_path / "corpus.db")
    _create_corpus_db(corpus_path)

    scorer = TDAScorer()
    scorer.configure({"signal_db": "/nonexistent/path.db"})

    corpus_conn = sqlite3.connect(corpus_path)
    corpus_conn.row_factory = sqlite3.Row
    scorer.precompute(corpus_conn)

    results = scorer.retrieve("query", corpus_conn, limit=10)
    corpus_conn.close()
    assert results == []


def test_limit_respected(scorer_setup):
    scorer, conn = scorer_setup
    results = scorer.retrieve("test query", conn, limit=2)
    assert len(results) <= 2
