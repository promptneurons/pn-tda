"""Adapter for Open Planter obsidian-refs.json output."""

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

from pn_tda.adapters.base import Graph
from pn_tda.utils.geometry import jaccard_distance


class ObsidianRefsGraph(Graph):
    """Graph constructed from obsidian-refs.json.

    Expected JSON format:
    {
      "nodes": [{"id": "...", "type": "file|gln|term", ...}],
      "edges": [{"source": "...", "target": "...", "type": "..."}]
    }

    Uses Jaccard distance on neighbor sets to capture semantic
    similarity through co-occurrence.
    """

    def __init__(self, refs_path: str):
        with open(refs_path) as f:
            data = json.load(f)
        self._load(data)

    def _load(self, data: dict) -> None:
        # Index nodes by ID
        self._node_attrs: dict[str, dict[str, Any]] = {}
        for node in data.get("nodes", []):
            node_id = node["id"]
            self._node_attrs[node_id] = dict(node)

        # Build adjacency lists (undirected) and edge list
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._edge_list: list[Tuple[str, str]] = []
        seen_edges: set[tuple[str, str]] = set()

        for edge in data.get("edges", []):
            src = edge["source"]
            tgt = edge["target"]
            # Skip edges referencing unknown nodes
            if src not in self._node_attrs or tgt not in self._node_attrs:
                continue
            # Deduplicate undirected
            canonical = (min(src, tgt), max(src, tgt))
            if canonical not in seen_edges:
                seen_edges.add(canonical)
                self._edge_list.append((src, tgt))
            self._adjacency[src].add(tgt)
            self._adjacency[tgt].add(src)

    def nodes(self) -> Iterable[str]:
        return list(self._node_attrs.keys())

    def edges(self) -> Iterable[Tuple[str, str]]:
        return self._edge_list

    def get_distance(self, u: str, v: str) -> float:
        """Jaccard distance on neighbor sets."""
        neighbors_u = self._adjacency.get(u, set())
        neighbors_v = self._adjacency.get(v, set())
        if not neighbors_u and not neighbors_v:
            return 1.0
        return jaccard_distance(neighbors_u, neighbors_v)

    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self._node_attrs:
            raise KeyError(f"Unknown node: {node_id}")
        return dict(self._node_attrs[node_id])

    def get_neighbors(self, node_id: str) -> Iterable[str]:
        return list(self._adjacency.get(node_id, set()))
