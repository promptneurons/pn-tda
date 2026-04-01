"""Tests for SignalDBGraph adapter."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from pn_tda.adapters.signal_db import SignalDBGraph


@pytest.fixture
def signal_db_path():
    """Create a fixture SignalDB SQLite database with known data."""
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
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                signal_type TEXT NOT NULL,
                signal_value TEXT NOT NULL,
                line_number INTEGER,
                context TEXT,
                UNIQUE(doc_id, signal_type, signal_value, line_number)
            );

            CREATE TABLE edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                target_path TEXT NOT NULL,
                target_doc_id TEXT,
                edge_type TEXT NOT NULL,
                line_number INTEGER,
                UNIQUE(source_doc_id, target_path, edge_type, line_number)
            );

            -- 4 documents across 2 sprints
            INSERT INTO documents VALUES
                ('d1', 'Sprint/26033/note-a.md', 'note-a.md', '26033', 2026, 3, 3, 0, '030959TMAR26', 'FY2026'),
                ('d2', 'Sprint/26033/note-b.md', 'note-b.md', '26033', 2026, 3, 3, 1, '031045TMAR26', 'FY2026'),
                ('d3', 'Sprint/26033/note-c.md', 'note-c.md', '26033', 2026, 3, 3, 0, NULL, 'FY2026'),
                ('d4', 'Sprint/26034/note-d.md', 'note-d.md', '26034', 2026, 3, 4, 0, '100800TMAR26', 'FY2026');

            -- Signals: d1 and d2 share classification refs, d3 is isolated
            INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
                ('d1', 'classification_ref', 'CEC B512', 5),
                ('d1', 'classification_ref', 'JDN 3-01', 10),
                ('d1', 'dtg_content', '030959TMAR26', 1),
                ('d2', 'classification_ref', 'CEC B512', 3),
                ('d2', 'classification_ref', 'DDC 332.6', 7),
                ('d2', 'dtg_content', '031045TMAR26', 1),
                ('d3', 'classification_ref', 'SUMO Process', 2),
                ('d4', 'classification_ref', 'CEC B512', 4),
                ('d4', 'classification_ref', 'JDN 3-01', 8);

            -- Edges: d1→d2 (wiki_link), d2→d3 (uplink), d1→external (unresolved)
            INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
                ('d1', 'Sprint/26033/note-b.md', 'd2', 'wiki_link', 12),
                ('d2', 'Sprint/26033/note-c.md', 'd3', 'uplink', 5),
                ('d1', 'https://dev.azure.com/org/project', NULL, 'azure_devops_link', 20),
                ('d4', 'Sprint/26033/note-a.md', 'd1', 'wiki_link', 3);
        """)
        conn.commit()
        conn.close()
        yield db_path


def test_nodes(signal_db_path):
    g = SignalDBGraph(signal_db_path)
    nodes = list(g.nodes())
    assert len(nodes) == 4
    assert set(nodes) == {"d1", "d2", "d3", "d4"}


def test_edges_only_resolved(signal_db_path):
    """External links (target_doc_id IS NULL) should be excluded."""
    g = SignalDBGraph(signal_db_path)
    edges = list(g.edges())
    # 3 resolved edges: d1→d2, d2→d3, d4→d1
    assert len(edges) == 3
    # External azure_devops_link should not appear
    edge_nodes = {n for e in edges for n in e}
    assert edge_nodes <= {"d1", "d2", "d3", "d4"}


def test_neighbors_undirected(signal_db_path):
    """Adjacency should be undirected (both directions)."""
    g = SignalDBGraph(signal_db_path)
    # d1→d2 edge means d2 is neighbor of d1 AND d1 is neighbor of d2
    assert "d2" in list(g.get_neighbors("d1"))
    assert "d1" in list(g.get_neighbors("d2"))
    # d4→d1 edge
    assert "d4" in list(g.get_neighbors("d1"))
    assert "d1" in list(g.get_neighbors("d4"))


def test_distance_symmetry(signal_db_path):
    g = SignalDBGraph(signal_db_path)
    assert g.get_distance("d1", "d2") == g.get_distance("d2", "d1")
    assert g.get_distance("d1", "d3") == g.get_distance("d3", "d1")


def test_distance_range(signal_db_path):
    """All distances should be in [0, 1]."""
    g = SignalDBGraph(signal_db_path)
    nodes = list(g.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            d = g.get_distance(u, v)
            assert 0.0 <= d <= 1.0, f"Distance {u}-{v} = {d} out of range"


def test_distance_shared_signals_closer(signal_db_path):
    """d1 and d4 share CEC B512 + JDN 3-01, so they should be closer than d1 and d3."""
    g = SignalDBGraph(signal_db_path)
    # d1 and d4 share two classification refs
    # d1 and d3 share no classification refs
    dist_d1_d4 = g.get_distance("d1", "d4")
    dist_d1_d3 = g.get_distance("d1", "d3")
    assert dist_d1_d4 < dist_d1_d3


def test_node_attributes(signal_db_path):
    g = SignalDBGraph(signal_db_path)
    attrs = g.get_node_attributes("d2")
    assert attrs["sprint_id"] == "26033"
    assert attrs["is_focus_hub"] == 1
    assert attrs["filename"] == "note-b.md"


def test_unknown_node_raises(signal_db_path):
    g = SignalDBGraph(signal_db_path)
    with pytest.raises(KeyError):
        g.get_node_attributes("nonexistent")
