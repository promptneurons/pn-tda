"""Tests for ObsidianRefsGraph adapter."""

import json
import tempfile
from pathlib import Path

import pytest

from pn_tda.adapters.obsidian_refs import ObsidianRefsGraph


SAMPLE_REFS = {
    "nodes": [
        {"id": "f1", "type": "file", "path": "notes/topic-a.md", "label": "Topic A"},
        {"id": "f2", "type": "file", "path": "notes/topic-b.md", "label": "Topic B"},
        {"id": "f3", "type": "file", "path": "notes/topic-c.md", "label": "Topic C"},
        {"id": "g1", "type": "gln", "value": "1a1b", "label": "GLN 1a1b"},
        {"id": "g2", "type": "gln", "value": "1a1c", "label": "GLN 1a1c"},
        {"id": "t1", "type": "term", "value": "persistent homology", "label": "Persistent Homology"},
    ],
    "edges": [
        {"source": "f1", "target": "f2", "type": "wikilink"},
        {"source": "f1", "target": "f3", "type": "wikilink"},
        {"source": "f2", "target": "f3", "type": "wikilink"},
        {"source": "f1", "target": "g1", "type": "references_gln"},
        {"source": "f2", "target": "g1", "type": "references_gln"},
        {"source": "f3", "target": "g2", "type": "references_gln"},
        {"source": "f1", "target": "t1", "type": "mentions"},
        {"source": "f2", "target": "t1", "type": "mentions"},
    ],
}


@pytest.fixture
def refs_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "obsidian-refs.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_REFS, f)
        yield path


def test_nodes(refs_path):
    g = ObsidianRefsGraph(refs_path)
    nodes = list(g.nodes())
    assert len(nodes) == 6
    assert set(nodes) == {"f1", "f2", "f3", "g1", "g2", "t1"}


def test_edges(refs_path):
    g = ObsidianRefsGraph(refs_path)
    edges = list(g.edges())
    assert len(edges) == 8


def test_neighbors_undirected(refs_path):
    """Edges are treated as undirected."""
    g = ObsidianRefsGraph(refs_path)
    # f1→g1 means g1 is neighbor of f1 and f1 is neighbor of g1
    assert "g1" in list(g.get_neighbors("f1"))
    assert "f1" in list(g.get_neighbors("g1"))


def test_neighbor_counts(refs_path):
    """Verify specific neighbor counts from the fixture."""
    g = ObsidianRefsGraph(refs_path)
    # f1: f2, f3, g1, t1 = 4 neighbors
    assert len(list(g.get_neighbors("f1"))) == 4
    # g1: f1, f2 = 2 neighbors
    assert len(list(g.get_neighbors("g1"))) == 2
    # g2: f3 = 1 neighbor
    assert len(list(g.get_neighbors("g2"))) == 1


def test_distance_jaccard_correctness(refs_path):
    """Verify Jaccard distance on neighbor sets."""
    g = ObsidianRefsGraph(refs_path)
    # f1 neighbors: {f2, f3, g1, t1}
    # f2 neighbors: {f1, f3, g1, t1}
    # Intersection: {f3, g1, t1} = 3
    # Union: {f1, f2, f3, g1, t1} = 5
    # Jaccard similarity = 3/5, distance = 1 - 3/5 = 0.4
    dist = g.get_distance("f1", "f2")
    assert abs(dist - 0.4) < 1e-9


def test_distance_symmetry(refs_path):
    g = ObsidianRefsGraph(refs_path)
    assert g.get_distance("f1", "g2") == g.get_distance("g2", "f1")


def test_distance_range(refs_path):
    g = ObsidianRefsGraph(refs_path)
    nodes = list(g.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            d = g.get_distance(u, v)
            assert 0.0 <= d <= 1.0


def test_distance_isolated_nodes_max(refs_path):
    """Nodes with no shared neighbors should have distance 1.0."""
    g = ObsidianRefsGraph(refs_path)
    # g2 neighbors: {f3}
    # t1 neighbors: {f1, f2}
    # Intersection: {} = 0
    # Distance = 1.0
    assert g.get_distance("g2", "t1") == 1.0


def test_node_attributes(refs_path):
    g = ObsidianRefsGraph(refs_path)
    attrs = g.get_node_attributes("f1")
    assert attrs["type"] == "file"
    assert attrs["path"] == "notes/topic-a.md"
    assert attrs["label"] == "Topic A"

    attrs_gln = g.get_node_attributes("g1")
    assert attrs_gln["type"] == "gln"
    assert attrs_gln["value"] == "1a1b"


def test_unknown_node_raises(refs_path):
    g = ObsidianRefsGraph(refs_path)
    with pytest.raises(KeyError):
        g.get_node_attributes("nonexistent")


def test_edges_skip_unknown_nodes():
    """Edges referencing nodes not in the nodes list should be skipped."""
    data = {
        "nodes": [{"id": "a", "type": "file"}, {"id": "b", "type": "file"}],
        "edges": [
            {"source": "a", "target": "b", "type": "wikilink"},
            {"source": "a", "target": "ghost", "type": "wikilink"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "refs.json")
        with open(path, "w") as f:
            json.dump(data, f)
        g = ObsidianRefsGraph(path)
        assert len(list(g.edges())) == 1
