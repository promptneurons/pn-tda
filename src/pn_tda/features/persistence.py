"""Persistence diagram feature extraction."""

from __future__ import annotations

import math

from pn_tda.core.persistence import PersistenceInterval


class PersistenceFeatureExtractor:
    """Extract scalar features from persistence diagrams.

    Per dimension: count, total_persistence, mean_persistence,
    max_persistence, entropy.
    Global: total_intervals, infinite_intervals.
    """

    def extract(
        self,
        intervals: list[PersistenceInterval],
        max_dimension: int = 2,
    ) -> dict:
        """Extract scalar features from persistence intervals.

        Returns:
            {
                "dim_0_count": int,
                "dim_0_total_persistence": float,
                "dim_0_mean_persistence": float,
                "dim_0_max_persistence": float,
                "dim_0_entropy": float,
                ...  (for each dimension)
                "total_intervals": int,
                "infinite_intervals": int,
            }
        """
        result: dict = {}

        for d in range(max_dimension + 1):
            finite = [
                iv for iv in intervals
                if iv.dimension == d and iv.death != float("inf")
            ]
            persistences = [iv.persistence for iv in finite]

            prefix = f"dim_{d}"
            result[f"{prefix}_count"] = len(finite)

            if persistences:
                total = sum(persistences)
                result[f"{prefix}_total_persistence"] = total
                result[f"{prefix}_mean_persistence"] = total / len(persistences)
                result[f"{prefix}_max_persistence"] = max(persistences)
                result[f"{prefix}_entropy"] = _persistence_entropy(persistences)
            else:
                result[f"{prefix}_total_persistence"] = 0.0
                result[f"{prefix}_mean_persistence"] = 0.0
                result[f"{prefix}_max_persistence"] = 0.0
                result[f"{prefix}_entropy"] = 0.0

        result["total_intervals"] = len(intervals)
        result["infinite_intervals"] = sum(
            1 for iv in intervals if iv.death == float("inf")
        )

        return result


def _persistence_entropy(persistences: list[float]) -> float:
    """Compute persistence entropy: -Σ (p_i * log(p_i)).

    Where p_i = persistence_i / total_persistence.
    Returns 0.0 for empty or single-element lists.
    """
    if len(persistences) <= 1:
        return 0.0
    total = sum(persistences)
    if total == 0:
        return 0.0
    entropy = 0.0
    for p in persistences:
        if p > 0:
            prob = p / total
            entropy -= prob * math.log(prob)
    return entropy
