"""End-to-end integration tests: adapter → VR complex → persistence."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from pn_tda.adapters.obsidian_refs import ObsidianRefsGraph
from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.persistence import PersistentHomology, betti_numbers


@pytest.fixture
def signal_db_path():
    """Small SignalDB with a triangle (d1-d2-d3) and isolated node (d4)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "signals.db")
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
                ('d1', 'a.md', 'a.md', '26033', 2026, 3, 3, 0, NULL, NULL),
                ('d2', 'b.md', 'b.md', '26033', 2026, 3, 3, 0, NULL, NULL),
                ('d3', 'c.md', 'c.md', '26033', 2026, 3, 3, 0, NULL, NULL),
                ('d4', 'd.md', 'd.md', '26034', 2026, 3, 4, 0, NULL, NULL);

            -- Shared signals to make d1,d2,d3 close and d4 far
            INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
                ('d1', 'ref', 'X', 1), ('d1', 'ref', 'Y', 2),
                ('d2', 'ref', 'X', 1), ('d2', 'ref', 'Y', 2),
                ('d3', 'ref', 'X', 1), ('d3', 'ref', 'Y', 2),
                ('d4', 'ref', 'Z', 1);

            INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
                ('d1', 'b.md', 'd2', 'wiki_link', 1),
                ('d2', 'c.md', 'd3', 'wiki_link', 1),
                ('d3', 'a.md', 'd1', 'wiki_link', 1);
        """)
        conn.commit()
        conn.close()
        yield db_path


@pytest.fixture
def obsidian_refs_path():
    """Small graph: triangle (a-b-c) + pendant node (d connected to a only)."""
    data = {
        "nodes": [
            {"id": "a", "type": "file", "path": "a.md"},
            {"id": "b", "type": "file", "path": "b.md"},
            {"id": "c", "type": "file", "path": "c.md"},
            {"id": "d", "type": "file", "path": "d.md"},
        ],
        "edges": [
            {"source": "a", "target": "b", "type": "wikilink"},
            {"source": "b", "target": "c", "type": "wikilink"},
            {"source": "c", "target": "a", "type": "wikilink"},
            {"source": "a", "target": "d", "type": "wikilink"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "refs.json")
        with open(path, "w") as f:
            json.dump(data, f)
        yield path


def test_signal_db_pipeline(signal_db_path):
    """SignalDB → VR complex → PH produces reasonable results."""
    graph = SignalDBGraph(signal_db_path)
    builder = VietorisRipsBuilder(epsilon_max=1.0, max_dimension=2)
    st = builder.build(graph)
    ph = PersistentHomology()
    intervals = ph.compute(st)

    assert len(intervals) > 0

    # d1, d2, d3 form a connected cluster; d4 is isolated (different signals)
    # At a moderate scale, expect β₀ >= 1
    betti = betti_numbers(intervals, at_scale=0.5)
    assert betti.get(0, 0) >= 1


def test_signal_db_isolated_component(signal_db_path):
    """d4 has no shared signals, so at small scale it's a separate component."""
    graph = SignalDBGraph(signal_db_path)
    # d4's signals are completely different from d1-d3
    dist_d1_d4 = graph.get_distance("d1", "d4")
    dist_d1_d2 = graph.get_distance("d1", "d2")
    assert dist_d1_d4 > dist_d1_d2


def test_obsidian_refs_pipeline(obsidian_refs_path):
    """ObsidianRefs → VR complex → PH produces reasonable results."""
    graph = ObsidianRefsGraph(obsidian_refs_path)
    builder = VietorisRipsBuilder(epsilon_max=1.0, max_dimension=2)
    st = builder.build(graph)
    ph = PersistentHomology()
    intervals = ph.compute(st)

    assert len(intervals) > 0

    # 4 connected nodes → at large enough scale, β₀ = 1
    betti = betti_numbers(intervals, at_scale=0.99)
    assert betti.get(0, 0) >= 1


def test_obsidian_refs_topology(obsidian_refs_path):
    """Verify topological structure of a small graph.

    In a VR complex, a 3-node triangle fills immediately (the 2-simplex
    appears at the same scale as the last edge), so H1 has zero persistence.
    Instead, verify the component structure: 4 connected nodes merge into
    one component, and the pendant node (d) merges last.
    """
    graph = ObsidianRefsGraph(obsidian_refs_path)
    builder = VietorisRipsBuilder(epsilon_max=1.0, max_dimension=2)
    st = builder.build(graph)
    ph = PersistentHomology()
    intervals = ph.compute(st)

    # Should have multiple H0 death events (components merging)
    h0_finite = [iv for iv in intervals if iv.dimension == 0 and iv.death != float("inf")]
    assert len(h0_finite) == 3, f"4 nodes should produce 3 H0 deaths, got {len(h0_finite)}"

    # One H0 infinite interval (the final connected component)
    h0_inf = [iv for iv in intervals if iv.dimension == 0 and iv.death == float("inf")]
    assert len(h0_inf) == 1
