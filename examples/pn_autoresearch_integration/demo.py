#!/usr/bin/env python3
"""Demo: TDA Scorer standalone execution.

Creates small in-memory databases (corpus + signals), runs the TDA scorer
pipeline, and prints scored results with feature details.

Usage:
    cd pn-tda
    python examples/pn_autoresearch_integration/demo.py
"""

import sqlite3
import sys
import tempfile
from pathlib import Path


def create_signal_db(db_path: str) -> None:
    """Create a small signal DB with 5 connected documents."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            sprint_id TEXT,
            sprint_year INTEGER,
            sprint_month INTEGER,
            sprint_in_month INTEGER,
            is_focus_hub BOOLEAN DEFAULT 0,
            title_dtg TEXT,
            parent_group TEXT
        );
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            signal_value TEXT NOT NULL,
            line_number INTEGER,
            context TEXT,
            UNIQUE(doc_id, signal_type, signal_value, line_number)
        );
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_doc_id TEXT NOT NULL,
            target_path TEXT NOT NULL,
            target_doc_id TEXT,
            edge_type TEXT NOT NULL,
            line_number INTEGER,
            UNIQUE(source_doc_id, target_path, edge_type, line_number)
        );

        INSERT INTO documents VALUES
            ('d1', 'notes/tda-intro.md',    'tda-intro.md',    '26033', 2026, 3, 3, 1, NULL, NULL),
            ('d2', 'notes/homology.md',     'homology.md',     '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d3', 'notes/filtration.md',   'filtration.md',   '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d4', 'notes/persistence.md',  'persistence.md',  '26033', 2026, 3, 3, 0, NULL, NULL),
            ('d5', 'notes/applications.md', 'applications.md', '26033', 2026, 3, 3, 0, NULL, NULL);

        INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
            ('d1', 'classification_ref', 'CEC B512', 1),
            ('d1', 'classification_ref', 'SUMO Process', 5),
            ('d2', 'classification_ref', 'CEC B512', 2),
            ('d2', 'classification_ref', 'CEC B514', 4),
            ('d3', 'classification_ref', 'CEC B512', 3),
            ('d3', 'classification_ref', 'SUMO Process', 6),
            ('d4', 'classification_ref', 'CEC B514', 1),
            ('d4', 'classification_ref', 'DDC 510.0', 3),
            ('d5', 'classification_ref', 'SUMO Process', 2);

        INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
            ('d1', 'notes/homology.md',    'd2', 'wiki_link', 3),
            ('d1', 'notes/filtration.md',  'd3', 'wiki_link', 5),
            ('d2', 'notes/persistence.md', 'd4', 'wiki_link', 7),
            ('d3', 'notes/tda-intro.md',   'd1', 'uplink', 1),
            ('d4', 'notes/tda-intro.md',   'd1', 'uplink', 1),
            ('d5', 'notes/filtration.md',  'd3', 'wiki_link', 4);
    """)
    conn.commit()
    conn.close()


def create_corpus_db(db_path: str) -> None:
    """Create a small corpus DB with chunks mapped to signal DB documents."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            content TEXT NOT NULL,
            heading TEXT
        );

        INSERT INTO documents VALUES
            ('d1', 'notes/tda-intro.md'),
            ('d2', 'notes/homology.md'),
            ('d3', 'notes/filtration.md'),
            ('d4', 'notes/persistence.md'),
            ('d5', 'notes/applications.md');

        INSERT INTO chunks VALUES
            ('c1a', 'd1', 'Introduction to topological data analysis and persistent homology.', 'TDA Overview'),
            ('c1b', 'd1', 'Vietoris-Rips complexes capture multi-scale structure.', 'VR Complexes'),
            ('c2a', 'd2', 'Homology groups classify topological spaces by their holes.', 'Homology Basics'),
            ('c3a', 'd3', 'A filtration is a nested sequence of simplicial complexes.', 'Filtrations'),
            ('c3b', 'd3', 'Sublevel set filtrations are commonly used in TDA.', 'Sublevel Sets'),
            ('c4a', 'd4', 'Persistent homology tracks birth and death of topological features.', 'Persistence'),
            ('c5a', 'd5', 'TDA applications include shape analysis and sensor networks.', 'Applications');
    """)
    conn.commit()
    conn.close()


def main():
    from examples.pn_autoresearch_integration.tda_scorer import TDAScorer

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path = str(Path(tmpdir) / "signals.db")
        corpus_path = str(Path(tmpdir) / "corpus.db")

        print("=== TDA Scorer Demo ===\n")

        # Create databases
        print("1. Creating signal database (5 docs, signals, edges)...")
        create_signal_db(signal_path)

        print("2. Creating corpus database (5 docs, 7 chunks)...")
        create_corpus_db(corpus_path)

        # Configure and precompute
        print("3. Configuring TDA scorer...")
        scorer = TDAScorer()
        scorer.configure({
            "signal_db": signal_path,
            "epsilon_max": 1.0,
            "max_dimension": 2,
            "num_scales": 10,
            "maturity_weight": 0.5,
            "persistence_weight": 0.3,
            "betti_weight": 0.2,
        })

        print("4. Running TDA pipeline (VR complex → PH → features)...")
        corpus_conn = sqlite3.connect(corpus_path)
        corpus_conn.row_factory = sqlite3.Row
        scorer.precompute(corpus_conn)

        # Show features
        features = scorer.get_features()
        print("\n--- TDA Features (corpus-level) ---")
        if features:
            sample = next(iter(features.values()))
            for key, val in sample.items():
                print(f"  {key}: {val:.4f}")
        else:
            print("  (no features computed)")

        # Retrieve scored chunks
        print("\n5. Scoring chunks...")
        results = scorer.retrieve("topological data analysis", corpus_conn, limit=10)
        corpus_conn.close()

        print(f"\n--- Scored Chunks ({len(results)} results) ---")
        for i, sc in enumerate(results, 1):
            print(f"  {i}. {sc.chunk_id}  score={sc.score:.4f}")

        print("\nDone.")


if __name__ == "__main__":
    # Allow running from project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
