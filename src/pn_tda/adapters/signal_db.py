"""Adapter for azure-wiki-analysis SignalDB SQLite output."""

import sqlite3
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

from pn_tda.adapters.base import Graph
from pn_tda.utils.geometry import jaccard_distance


class SignalDBGraph(Graph):
    """Graph constructed from azure-wiki-analysis SQLite database.

    Reads documents, signals, and edges tables. Nodes are documents
    (keyed by doc_id). Edges are resolved inter-document links.
    Distance is Jaccard distance on per-document signal sets.
    """

    def __init__(self, db_path: str):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            self._load(conn)
        finally:
            conn.close()

    def _load(self, conn: sqlite3.Connection) -> None:
        # Load documents
        self._doc_attrs: dict[str, dict[str, Any]] = {}
        for row in conn.execute("SELECT * FROM documents"):
            doc_id = row["doc_id"]
            self._doc_attrs[doc_id] = dict(row)

        # Load resolved edges (skip external links where target_doc_id is NULL)
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._edge_list: list[Tuple[str, str]] = []
        for row in conn.execute(
            "SELECT source_doc_id, target_doc_id FROM edges "
            "WHERE target_doc_id IS NOT NULL"
        ):
            src, tgt = row["source_doc_id"], row["target_doc_id"]
            # Only include edges between known documents
            if src in self._doc_attrs and tgt in self._doc_attrs:
                if (src, tgt) not in self._edge_set_check(src, tgt):
                    self._edge_list.append((src, tgt))
                # Undirected adjacency for TDA
                self._adjacency[src].add(tgt)
                self._adjacency[tgt].add(src)

        # Load signals per document for distance computation
        self._signals: dict[str, set[str]] = defaultdict(set)
        for row in conn.execute("SELECT doc_id, signal_type, signal_value FROM signals"):
            doc_id = row["doc_id"]
            if doc_id in self._doc_attrs:
                self._signals[doc_id].add(f"{row['signal_type']}:{row['signal_value']}")

        # Also add edge types as signals for richer distance computation
        for row in conn.execute(
            "SELECT source_doc_id, target_doc_id, edge_type FROM edges "
            "WHERE target_doc_id IS NOT NULL"
        ):
            src, tgt = row["source_doc_id"], row["target_doc_id"]
            edge_type = row["edge_type"]
            if src in self._doc_attrs:
                self._signals[src].add(f"edge:{edge_type}:{tgt}")
            if tgt in self._doc_attrs:
                self._signals[tgt].add(f"edge:{edge_type}:{src}")

    def _edge_set_check(self, src: str, tgt: str) -> set:
        """Deduplicate directed edges into undirected edge list."""
        return {(src, tgt), (tgt, src)} & set(self._edge_list)

    def nodes(self) -> Iterable[str]:
        return list(self._doc_attrs.keys())

    def edges(self) -> Iterable[Tuple[str, str]]:
        return self._edge_list

    def get_distance(self, u: str, v: str) -> float:
        """Jaccard distance on per-document signal sets."""
        signals_u = self._signals.get(u, set())
        signals_v = self._signals.get(v, set())
        if not signals_u and not signals_v:
            return 1.0
        return jaccard_distance(signals_u, signals_v)

    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self._doc_attrs:
            raise KeyError(f"Unknown node: {node_id}")
        return dict(self._doc_attrs[node_id])

    def get_neighbors(self, node_id: str) -> Iterable[str]:
        return list(self._adjacency.get(node_id, set()))
