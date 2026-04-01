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

        # Build path → doc_id index for resolving unresolved edges.
        # Wiki links use slugs (e.g. "Nomad-Focus") while source_path uses
        # filesystem paths ("Container/Analysis.md"), so we index multiple forms.
        path_to_doc: dict[str, str] = {}
        for doc_id, attrs in self._doc_attrs.items():
            sp = attrs.get("source_path", "")
            if not sp:
                continue
            path_to_doc[sp] = doc_id
            parts = sp.replace("\\", "/").split("/")
            if parts:
                fname = parts[-1]
                path_to_doc[fname] = doc_id  # "Analysis.md"
                # Stem without extension (wiki slug match)
                stem = fname.rsplit(".", 1)[0] if "." in fname else fname
                path_to_doc[stem] = doc_id  # "Analysis"
            if len(parts) >= 2:
                path_to_doc["/".join(parts[-2:])] = doc_id  # "Container/Analysis.md"
                # parent/stem slug
                stem2 = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
                path_to_doc[f"{parts[-2]}/{stem2}"] = doc_id  # "Container/Analysis"

        # Load edges — resolve target_doc_id from target_path if needed
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._edge_list: list[Tuple[str, str]] = []
        seen_edges: set[Tuple[str, str]] = set()
        for row in conn.execute(
            "SELECT source_doc_id, target_path, target_doc_id FROM edges"
        ):
            src = row["source_doc_id"]
            tgt = row["target_doc_id"]

            # Resolve unresolved edges by matching target_path
            if tgt is None:
                target_path = row["target_path"] or ""
                # Try exact match, then slug (stem without .md), then filename
                tgt = path_to_doc.get(target_path)
                if tgt is None:
                    stem = target_path.rsplit(".", 1)[0] if "." in target_path else target_path
                    tgt = path_to_doc.get(stem)
                if tgt is None:
                    fname = target_path.replace("\\", "/").split("/")[-1] if target_path else ""
                    tgt = path_to_doc.get(fname)
                    if tgt is None and "." in fname:
                        tgt = path_to_doc.get(fname.rsplit(".", 1)[0])
                if tgt is None:
                    continue  # truly external link

            if src not in self._doc_attrs or tgt not in self._doc_attrs:
                continue
            if src == tgt:
                continue  # skip self-loops
            canonical = (min(src, tgt), max(src, tgt))
            if canonical in seen_edges:
                continue
            seen_edges.add(canonical)
            self._edge_list.append((src, tgt))
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
