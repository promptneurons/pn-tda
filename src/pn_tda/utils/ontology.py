"""Ontology-aware distance metrics for TDA graph construction.

Combines three distance signals:
1. CEC/JDN hierarchy distance (Lowest Common Ancestor path length)
2. SUMO concept distance (WordNet → SUMO expansion, Jaccard)
3. Signal co-occurrence (existing Jaccard on raw signals)

The combined distance gives documents sharing an ontology parent
a shorter distance than flat token Jaccard alone.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

from pn_tda.utils.geometry import jaccard_distance

WORD_RE = re.compile(r"\b[a-z]{3,}\b")


class OntologyDistance:
    """Ontology-aware distance between documents.

    Loads concept hierarchy (CEC/JDN via concept_hierarchy table) and
    optionally SUMO/WordNet mappings. Computes a blended distance:

        d(u, v) = w_hier * hierarchy_dist + w_sumo * sumo_dist + w_signal * signal_dist

    where each component is in [0, 1].
    """

    def __init__(
        self,
        corpus_db_path: Optional[str] = None,
        wordnet_dir: Optional[str] = None,
        hierarchy_weight: float = 0.4,
        sumo_weight: float = 0.3,
        signal_weight: float = 0.3,
    ):
        self.hierarchy_weight = hierarchy_weight
        self.sumo_weight = sumo_weight
        self.signal_weight = signal_weight

        # Concept hierarchy: concept_uri → set of ancestor URIs
        self._ancestors: dict[str, set[str]] = {}
        # Chunk → concept URIs mapping (via concept_tags)
        self._doc_concepts: dict[str, set[str]] = defaultdict(set)
        # SUMO bridge
        self._sumo_index = None
        self._sumo_mappings = None
        # Doc → SUMO concepts
        self._doc_sumo: dict[str, frozenset[str]] = {}

        if corpus_db_path:
            self._load_hierarchy(corpus_db_path)
            self._load_doc_concepts(corpus_db_path)

        if wordnet_dir:
            self._load_sumo(wordnet_dir)

    def _load_hierarchy(self, db_path: str) -> None:
        """Load concept hierarchy and compute transitive ancestor sets."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Build parent map
        parent_of: dict[str, set[str]] = defaultdict(set)
        try:
            for row in conn.execute("SELECT child_uri, parent_uri FROM concept_hierarchy"):
                parent_of[row["child_uri"]].add(row["parent_uri"])
        except sqlite3.OperationalError:
            conn.close()
            return  # No hierarchy table

        # Compute transitive closure (ancestors)
        all_concepts = set(parent_of.keys())
        for row in conn.execute("SELECT concept_uri FROM concepts"):
            all_concepts.add(row["concept_uri"])

        for concept in all_concepts:
            ancestors = set()
            stack = [concept]
            while stack:
                c = stack.pop()
                for p in parent_of.get(c, set()):
                    if p not in ancestors:
                        ancestors.add(p)
                        stack.append(p)
            self._ancestors[concept] = ancestors

        conn.close()

    def _load_doc_concepts(self, db_path: str) -> None:
        """Load document → concept URI mappings from concept_tags."""
        conn = sqlite3.connect(db_path)
        try:
            # concept_tags links chunk_id → concept_uri
            # We need chunk_id → doc_id to aggregate per document
            rows = conn.execute(
                "SELECT ct.concept_uri, c.doc_id "
                "FROM concept_tags ct "
                "JOIN chunks c ON ct.chunk_id = c.chunk_id "
                "WHERE ct.confidence >= 0.5"
            ).fetchall()
            for row in rows:
                self._doc_concepts[row[1]].add(row[0])
        except sqlite3.OperationalError:
            pass  # No concept_tags table
        conn.close()

    def _load_sumo(self, wordnet_dir: str) -> None:
        """Load SUMO/WordNet bridge."""
        # Import from the monorepo's sumo_wordnet module
        import importlib.util
        # Try to find sumo_wordnet.py in known locations
        candidates = [
            Path(wordnet_dir).parent / "sumo_wordnet.py",
            Path(__file__).resolve().parents[3] / "data" / "Ontologies" / "sumo_wordnet.py",
        ]
        for candidate in candidates:
            if candidate.is_file():
                spec = importlib.util.spec_from_file_location("sumo_wordnet", str(candidate))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._sumo_index, self._sumo_mappings = mod.load_sumo_db(wordnet_dir)
                self._words_to_sumo = mod.words_to_sumo
                return

        # Fallback: try direct import
        try:
            from sumo_wordnet import load_sumo_db, words_to_sumo
            self._sumo_index, self._sumo_mappings = load_sumo_db(wordnet_dir)
            self._words_to_sumo = words_to_sumo
        except ImportError:
            pass  # SUMO not available

    def expand_doc_sumo(self, doc_id: str, content_words: list[str]) -> frozenset[str]:
        """Get or compute SUMO concepts for a document."""
        if doc_id in self._doc_sumo:
            return self._doc_sumo[doc_id]
        if self._sumo_index is None:
            return frozenset()
        concepts = self._words_to_sumo(content_words, self._sumo_index, self._sumo_mappings)
        self._doc_sumo[doc_id] = concepts
        return concepts

    def hierarchy_distance(self, doc_a: str, doc_b: str) -> float:
        """Distance based on concept hierarchy (LCA path length).

        Documents sharing a direct parent concept get low distance.
        Documents with no shared ancestors get distance 1.0.
        """
        concepts_a = self._doc_concepts.get(doc_a, set())
        concepts_b = self._doc_concepts.get(doc_b, set())

        if not concepts_a or not concepts_b:
            return 1.0

        # Collect all ancestors for each doc's concepts
        ancestors_a = set()
        for c in concepts_a:
            ancestors_a.add(c)
            ancestors_a.update(self._ancestors.get(c, set()))

        ancestors_b = set()
        for c in concepts_b:
            ancestors_b.add(c)
            ancestors_b.update(self._ancestors.get(c, set()))

        # Shared ancestors = documents are in the same ontology neighborhood
        shared = ancestors_a & ancestors_b
        if not shared:
            # Check direct concept overlap
            direct_shared = concepts_a & concepts_b
            if direct_shared:
                return 0.0  # Same concepts
            return 1.0

        # Distance inversely proportional to specificity of shared ancestor
        # More shared ancestors (especially specific ones) = closer
        all_ancestors = ancestors_a | ancestors_b
        return 1.0 - len(shared) / len(all_ancestors)

    def sumo_distance(
        self, doc_a: str, doc_b: str,
        words_a: list[str], words_b: list[str],
    ) -> float:
        """SUMO concept distance via WordNet expansion."""
        if self._sumo_index is None:
            return 1.0

        concepts_a = self.expand_doc_sumo(doc_a, words_a)
        concepts_b = self.expand_doc_sumo(doc_b, words_b)

        if not concepts_a or not concepts_b:
            return 1.0

        intersection = concepts_a & concepts_b
        union = concepts_a | concepts_b
        return 1.0 - len(intersection) / len(union)

    def combined_distance(
        self,
        doc_a: str, doc_b: str,
        signals_a: set[str], signals_b: set[str],
        words_a: list[str] | None = None,
        words_b: list[str] | None = None,
    ) -> float:
        """Blended ontology-aware distance.

        Combines hierarchy distance, SUMO distance, and signal Jaccard
        with configurable weights.
        """
        h_dist = self.hierarchy_distance(doc_a, doc_b)
        s_dist = jaccard_distance(signals_a, signals_b) if signals_a or signals_b else 1.0

        active_weight = self.hierarchy_weight + self.signal_weight
        weighted = self.hierarchy_weight * h_dist + self.signal_weight * s_dist

        # Add SUMO if available and words provided
        if self._sumo_index is not None and words_a is not None and words_b is not None:
            su_dist = self.sumo_distance(doc_a, doc_b, words_a, words_b)
            weighted += self.sumo_weight * su_dist
            active_weight += self.sumo_weight

        if active_weight == 0:
            return 1.0
        return weighted / active_weight
