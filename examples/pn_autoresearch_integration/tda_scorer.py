"""TDA Scorer for kitsap-searchengine-lite integration.

Self-contained scorer that uses pn-tda topological features to score
document chunks. Follows the Scorer interface from kitsap-searchengine-lite.

Can be used standalone (includes interface stubs) or integrated directly
by copying into kitsap-searchengine-lite/src/scorers/.
"""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.persistence import PersistentHomology
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor


# --- Scorer interface stubs (mirrors kitsap-searchengine-lite/src/scorer.py) ---
# When integrating into kitsap-searchengine-lite, replace these with:
#   from src.scorer import Scorer, ScoredChunk


@dataclass
class ScoredChunk:
    chunk_id: str
    score: float


class Scorer(ABC):
    name: str = ""

    @abstractmethod
    def retrieve(
        self, query: str, db: sqlite3.Connection, limit: int = 100
    ) -> list[ScoredChunk]:
        pass

    def precompute(self, db: sqlite3.Connection) -> None:
        pass

    def configure(self, params: dict) -> None:
        pass


# --- TDA Scorer Implementation ---


class TDAScorer(Scorer):
    """Score chunks using topological features of their source documents.

    Pre-computes TDA features (maturity, persistence, Betti stability) from
    a SignalDB, then scores each chunk based on its document's features.

    Configuration (via YAML or configure()):
        signal_db:        Path to azure-wiki-analysis SQLite database
        epsilon_max:      VR complex distance threshold (default: 0.7)
        max_dimension:    Maximum simplex dimension (default: 2)
        num_scales:       Number of filtration scales (default: 10)
        maturity_weight:  Weight for maturity_score (default: 0.5)
        persistence_weight: Weight for persistence entropy (default: 0.3)
        betti_weight:     Weight for Betti stability (default: 0.2)
    """

    name = "tda_features"

    def __init__(self):
        self.signal_db_path: str = ""
        self.epsilon_max: float = 0.7
        self.max_dimension: int = 2
        self.num_scales: int = 10
        self.maturity_weight: float = 0.5
        self.persistence_weight: float = 0.3
        self.betti_weight: float = 0.2

        # Cached per-doc features after precompute()
        self._doc_scores: dict[str, float] = {}
        self._doc_features: dict[str, dict] = {}
        self._ready = False

    def configure(self, params: dict) -> None:
        self.signal_db_path = params.get(
            "signal_db", os.environ.get("TDA_SIGNAL_DB", "")
        )
        self.epsilon_max = params.get("epsilon_max", 0.7)
        self.max_dimension = params.get("max_dimension", 2)
        self.num_scales = params.get("num_scales", 10)
        self.maturity_weight = params.get("maturity_weight", 0.5)
        self.persistence_weight = params.get("persistence_weight", 0.3)
        self.betti_weight = params.get("betti_weight", 0.2)

    def precompute(self, db: sqlite3.Connection) -> None:
        """Run TDA pipeline on SignalDB and cache per-document scores."""
        if not self.signal_db_path or not os.path.isfile(self.signal_db_path):
            return

        # Load graph from signal DB
        graph = SignalDBGraph(self.signal_db_path)
        doc_ids = list(graph.nodes())
        if not doc_ids:
            return

        # Build VR complex and compute persistent homology
        builder = VietorisRipsBuilder(
            epsilon_max=self.epsilon_max, max_dimension=self.max_dimension
        )
        st = builder.build(graph)
        intervals = PersistentHomology().compute(st)

        # Extract features
        betti_ext = BettiNumberExtractor()
        persist_ext = PersistenceFeatureExtractor()
        maturity_scorer = ThreadMaturityScorer()

        betti_result = betti_ext.extract(
            intervals,
            num_scales=self.num_scales,
            epsilon_max=self.epsilon_max,
            max_dimension=self.max_dimension,
        )
        persist_result = persist_ext.extract(
            intervals, max_dimension=self.max_dimension
        )
        maturity_result = maturity_scorer.score(
            intervals, graph, num_scales=self.num_scales, epsilon_max=self.epsilon_max
        )

        # Compute a single composite score for the whole corpus
        maturity_score = maturity_result.get("maturity_score", 0.0)

        # Persistence entropy (normalized to [0,1] using log(n) as max)
        import math

        dim0_entropy = persist_result.get("dim_0_entropy", 0.0)
        dim0_count = persist_result.get("dim_0_count", 1)
        max_entropy = math.log(max(dim0_count, 2))
        norm_entropy = min(dim0_entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0

        # Betti stability: 1 - normalized std of β₀ (low std = stable)
        betti_std = betti_result["summary"].get("betti_0_std", 0.0)
        betti_mean = betti_result["summary"].get("betti_0_mean", 1.0)
        betti_stability = max(
            0.0, 1.0 - (betti_std / betti_mean if betti_mean > 0 else 0.0)
        )

        # Weighted composite
        total_weight = self.maturity_weight + self.persistence_weight + self.betti_weight
        if total_weight == 0:
            total_weight = 1.0

        composite = (
            self.maturity_weight * maturity_score
            + self.persistence_weight * norm_entropy
            + self.betti_weight * betti_stability
        ) / total_weight

        # Store per-doc: all documents in the signal DB get the corpus-level score.
        # In a more advanced version, per-document neighborhood features would
        # differentiate individual documents.
        features = {
            "maturity_score": maturity_score,
            "persistence_entropy_norm": norm_entropy,
            "betti_stability": betti_stability,
            "composite": composite,
        }

        for doc_id in doc_ids:
            self._doc_scores[doc_id] = composite
            self._doc_features[doc_id] = features

        self._ready = True

    def retrieve(
        self, query: str, db: sqlite3.Connection, limit: int = 100
    ) -> list[ScoredChunk]:
        """Score chunks by their document's TDA features.

        All chunks from a document with TDA features receive the same score
        (topological features are document-level, not chunk-level).
        """
        if not self._ready:
            return []

        rows = db.execute("SELECT chunk_id, doc_id FROM chunks").fetchall()

        scored = []
        for row in rows:
            doc_id = row["doc_id"] if isinstance(row, sqlite3.Row) else row[1]
            chunk_id = row["chunk_id"] if isinstance(row, sqlite3.Row) else row[0]

            if doc_id in self._doc_scores:
                scored.append(
                    ScoredChunk(chunk_id=chunk_id, score=self._doc_scores[doc_id])
                )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def get_features(self) -> dict[str, dict]:
        """Return cached per-document TDA features (for inspection/debugging)."""
        return dict(self._doc_features)
