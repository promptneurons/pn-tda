#!/usr/bin/env python3
"""NDCG Validation Experiment: Baseline vs TDA-enhanced search.

Compares keyword-only search against keyword + TDA features to measure
whether topological features improve search quality (NDCG@10).

Usage:
    cd pn-tda
    python3 examples/pn_autoresearch_integration/experiment.py

Output: TSV-formatted results showing baseline vs TDA metrics.
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.pn_autoresearch_integration.evaluate import (
    evaluate_search_quality,
    load_queries,
)
from examples.pn_autoresearch_integration.tda_scorer import TDAScorer

WORD_RE = re.compile(r"\b[a-z]{3,}\b")


def create_signal_db(db_path: str) -> None:
    """Create signal DB with 10 documents across two topic clusters."""
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
            ('d1',  'notes/tda-intro.md',      'tda-intro.md',      '26033', 2026, 3, 3, 1, NULL, NULL),
            ('d2',  'notes/homology.md',       'homology.md',       '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d3',  'notes/filtration.md',     'filtration.md',     '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d4',  'notes/persistence.md',    'persistence.md',    '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d5',  'notes/applications.md',   'applications.md',   '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d6',  'notes/simplex-tree.md',   'simplex-tree.md',   '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d7',  'notes/betti-numbers.md',  'betti-numbers.md',  '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d8',  'notes/distance-metrics.md','distance-metrics.md','26033', 2026, 3, 3, 0, NULL, NULL),
            ('d9',  'notes/graph-theory.md',   'graph-theory.md',   '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d10', 'notes/maturity.md',       'maturity.md',       '26033', 2026, 3, 3, 0, NULL, NULL);

        INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
            ('d1', 'classification_ref', 'CEC B512', 1),
            ('d1', 'classification_ref', 'SUMO Process', 3),
            ('d2', 'classification_ref', 'CEC B512', 2),
            ('d2', 'classification_ref', 'CEC B514', 4),
            ('d3', 'classification_ref', 'CEC B512', 1),
            ('d3', 'classification_ref', 'SUMO Process', 5),
            ('d4', 'classification_ref', 'CEC B514', 2),
            ('d4', 'classification_ref', 'DDC 510.0', 3),
            ('d5', 'classification_ref', 'SUMO Process', 1),
            ('d5', 'classification_ref', 'DDC 510.0', 4),
            ('d6', 'classification_ref', 'CEC B512', 2),
            ('d7', 'classification_ref', 'CEC B514', 1),
            ('d7', 'classification_ref', 'SUMO Process', 3),
            ('d8', 'classification_ref', 'DDC 510.0', 2),
            ('d9', 'classification_ref', 'CEC B512', 1),
            ('d9', 'classification_ref', 'DDC 510.0', 3),
            ('d10', 'classification_ref', 'SUMO Process', 1),
            ('d10', 'classification_ref', 'DDC 510.0', 4);

        INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
            ('d1', 'notes/homology.md',    'd2', 'wiki_link', 3),
            ('d1', 'notes/filtration.md',  'd3', 'wiki_link', 5),
            ('d1', 'notes/persistence.md', 'd4', 'wiki_link', 7),
            ('d2', 'notes/persistence.md', 'd4', 'wiki_link', 4),
            ('d2', 'notes/betti-numbers.md','d7', 'wiki_link', 8),
            ('d3', 'notes/tda-intro.md',   'd1', 'uplink', 1),
            ('d3', 'notes/simplex-tree.md','d6', 'wiki_link', 6),
            ('d4', 'notes/tda-intro.md',   'd1', 'uplink', 1),
            ('d5', 'notes/tda-intro.md',   'd1', 'wiki_link', 3),
            ('d5', 'notes/maturity.md',    'd10', 'wiki_link', 5),
            ('d6', 'notes/filtration.md',  'd3', 'wiki_link', 4),
            ('d7', 'notes/homology.md',    'd2', 'uplink', 1),
            ('d8', 'notes/distance-metrics.md','d8', 'wiki_link', 2),
            ('d9', 'notes/tda-intro.md',   'd1', 'wiki_link', 5),
            ('d10', 'notes/applications.md','d5', 'wiki_link', 3);
    """)
    conn.commit()
    conn.close()


def create_corpus_db(db_path: str) -> None:
    """Create corpus DB with chunks for the 10-document corpus."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, source_path TEXT NOT NULL);
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
            content TEXT NOT NULL, heading TEXT
        );

        INSERT INTO documents VALUES
            ('d1',  'notes/tda-intro.md'),
            ('d2',  'notes/homology.md'),
            ('d3',  'notes/filtration.md'),
            ('d4',  'notes/persistence.md'),
            ('d5',  'notes/applications.md'),
            ('d6',  'notes/simplex-tree.md'),
            ('d7',  'notes/betti-numbers.md'),
            ('d8',  'notes/distance-metrics.md'),
            ('d9',  'notes/graph-theory.md'),
            ('d10', 'notes/maturity.md');

        INSERT INTO chunks VALUES
            ('c1a', 'd1',  'Introduction to topological data analysis using persistent homology on point clouds and graphs.', 'TDA Overview'),
            ('c1b', 'd1',  'Vietoris-Rips complexes capture multi-scale topological structure from distance data.', 'VR Complexes'),
            ('c2a', 'd2',  'Homology groups classify topological spaces by counting holes of different dimensions.', 'Homology Basics'),
            ('c2b', 'd2',  'Betti numbers beta zero and beta one count connected components and loops respectively.', 'Betti Numbers'),
            ('c3a', 'd3',  'A filtration is a nested sequence of simplicial complexes indexed by a scale parameter.', 'Filtration Definition'),
            ('c3b', 'd3',  'Constructing a Vietoris-Rips filtration from pairwise distances with epsilon threshold.', 'VR Construction'),
            ('c4a', 'd4',  'Persistent homology tracks the birth and death of topological features across filtration scales.', 'Persistence Overview'),
            ('c4b', 'd4',  'Persistence diagrams represent intervals as points in the birth-death plane.', 'Persistence Diagrams'),
            ('c5a', 'd5',  'Applications of TDA include shape analysis, sensor networks, and knowledge graph structure.', 'TDA Applications'),
            ('c5b', 'd5',  'Thread maturity scoring uses topological features to measure research evolution.', 'Maturity Scoring'),
            ('c6a', 'd6',  'The simplex tree is an efficient trie-based data structure for storing simplicial complexes.', 'Simplex Tree'),
            ('c7a', 'd7',  'Betti numbers at multiple scales form feature vectors for topological machine learning.', 'Multi-scale Betti'),
            ('c8a', 'd8',  'Distance metrics including Jaccard, Euclidean, and cosine for graph construction.', 'Distance Metrics'),
            ('c9a', 'd9',  'Graph theory fundamentals: nodes, edges, adjacency, connected components, shortest paths.', 'Graph Basics'),
            ('c10a','d10', 'Maturity scoring combines connectedness, stability, persistence plateau, and dimensional shift.', 'Maturity Components');
    """)
    conn.commit()
    conn.close()


def keyword_jaccard(query: str, text: str) -> float:
    """Simple keyword Jaccard similarity."""
    q_tokens = set(WORD_RE.findall(query.lower()))
    t_tokens = set(WORD_RE.findall(text.lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens | t_tokens)


def make_baseline_search(corpus_conn: sqlite3.Connection):
    """Keyword Jaccard search only."""
    rows = corpus_conn.execute(
        "SELECT c.chunk_id, c.content, c.heading, d.source_path "
        "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id"
    ).fetchall()

    def search(query: str) -> list[str]:
        scored = []
        seen_paths = set()
        for r in rows:
            text = f"{r['heading'] or ''} {r['content']}"
            score = keyword_jaccard(query, text)
            path = r["source_path"]
            if score > 0 and path not in seen_paths:
                scored.append((path, score))
                seen_paths.add(path)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scored[:10]]

    return search


def make_tda_search(
    corpus_conn: sqlite3.Connection, tda_scorer: TDAScorer, tda_weight: float = 0.3
):
    """Keyword Jaccard + TDA composite score fusion."""
    rows = corpus_conn.execute(
        "SELECT c.chunk_id, c.doc_id, c.content, c.heading, d.source_path "
        "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id"
    ).fetchall()
    tda_features = tda_scorer.get_features()

    def search(query: str) -> list[str]:
        keyword_weight = 1.0 - tda_weight
        scored = []
        seen_paths = set()
        for r in rows:
            text = f"{r['heading'] or ''} {r['content']}"
            kw_score = keyword_jaccard(query, text)
            tda_score = tda_features.get(r["doc_id"], {}).get("composite", 0.0)
            combined = keyword_weight * kw_score + tda_weight * tda_score
            path = r["source_path"]
            if combined > 0 and path not in seen_paths:
                scored.append((path, combined))
                seen_paths.add(path)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scored[:10]]

    return search


def main():
    queries_path = str(Path(__file__).parent / "queries.jsonl")
    queries = load_queries(queries_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path = str(Path(tmpdir) / "signals.db")
        corpus_path = str(Path(tmpdir) / "corpus.db")

        print("=" * 60)
        print("NDCG Validation Experiment: Baseline vs TDA")
        print("=" * 60)

        # Create databases
        print("\n1. Creating databases (10 docs, 15 chunks, 10 queries)...")
        create_signal_db(signal_path)
        create_corpus_db(corpus_path)

        corpus_conn = sqlite3.connect(corpus_path)
        corpus_conn.row_factory = sqlite3.Row

        # --- Baseline ---
        print("\n2. Running BASELINE (keyword Jaccard only)...")
        t0 = time.time()
        baseline_search = make_baseline_search(corpus_conn)
        baseline_metrics = evaluate_search_quality(baseline_search, queries)
        baseline_time = time.time() - t0

        # --- TDA ---
        print("3. Running TDA pipeline (VR → PH → features)...")
        scorer = TDAScorer()
        scorer.configure({
            "signal_db": signal_path,
            "epsilon_max": 1.0,
            "max_dimension": 2,
            "num_scales": 10,
        })
        scorer.precompute(corpus_conn)

        print("4. Running TDA experiment (keyword + TDA fusion)...")
        t0 = time.time()
        tda_search = make_tda_search(corpus_conn, scorer, tda_weight=0.3)
        tda_metrics = evaluate_search_quality(tda_search, queries)
        tda_time = time.time() - t0

        corpus_conn.close()

        # --- Results ---
        delta_ndcg = tda_metrics["ndcg_at_10"] - baseline_metrics["ndcg_at_10"]
        if delta_ndcg > 0.01:
            decision = "KEEP"
        elif delta_ndcg < -0.02:
            decision = "DISCARD"
        else:
            decision = "NEUTRAL"

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\n{'Metric':<20} {'Baseline':>10} {'TDA':>10} {'Delta':>10}")
        print("-" * 52)
        print(
            f"{'NDCG@10':<20} {baseline_metrics['ndcg_at_10']:>10.4f} "
            f"{tda_metrics['ndcg_at_10']:>10.4f} {delta_ndcg:>+10.4f}"
        )
        print(
            f"{'MRR':<20} {baseline_metrics['mrr']:>10.4f} "
            f"{tda_metrics['mrr']:>10.4f} "
            f"{tda_metrics['mrr'] - baseline_metrics['mrr']:>+10.4f}"
        )
        print(
            f"{'P@3':<20} {baseline_metrics['precision_at_3']:>10.4f} "
            f"{tda_metrics['precision_at_3']:>10.4f} "
            f"{tda_metrics['precision_at_3'] - baseline_metrics['precision_at_3']:>+10.4f}"
        )
        print(f"{'Queries':<20} {baseline_metrics['num_queries']:>10d}")
        print(f"{'Eval time (s)':<20} {baseline_time:>10.3f} {tda_time:>10.3f}")
        print(f"\nΔNDCG@10 = {delta_ndcg:+.4f}  →  {decision}")

        # TSV output (pn-autoresearch compatible)
        print("\n--- TSV (for results.tsv) ---")
        print("run_tag\tconfig\tndcg_at_10\tmrr\tprecision_at_3\teval_seconds\tstatus\tdescription")
        print(
            f"baseline\tkeyword_only\t{baseline_metrics['ndcg_at_10']:.4f}\t"
            f"{baseline_metrics['mrr']:.4f}\t{baseline_metrics['precision_at_3']:.4f}\t"
            f"{baseline_time:.2f}\tBASELINE\tKeyword Jaccard only"
        )
        print(
            f"tda-exp\tkeyword+tda\t{tda_metrics['ndcg_at_10']:.4f}\t"
            f"{tda_metrics['mrr']:.4f}\t{tda_metrics['precision_at_3']:.4f}\t"
            f"{tda_time:.2f}\t{decision}\tKeyword 70% + TDA 30%"
        )


if __name__ == "__main__":
    main()
