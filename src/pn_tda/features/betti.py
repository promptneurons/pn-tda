"""Betti number extraction from persistence data."""

from __future__ import annotations

import math

from pn_tda.core.persistence import PersistenceInterval, betti_numbers


class BettiNumberExtractor:
    """Extract Betti number vectors at multiple filtration scales.

    Computes β_k at evenly-spaced scales and provides summary statistics.
    Output format matches the design doc JSON schema.
    """

    def extract(
        self,
        intervals: list[PersistenceInterval],
        num_scales: int = 10,
        epsilon_max: float = 1.0,
        max_dimension: int = 2,
    ) -> dict:
        """Extract Betti number vectors and summary statistics.

        Returns:
            {
                "scales": [0.0, 0.1, ...],
                "betti_0": [n, n, ...],
                "betti_1": [n, n, ...],
                "summary": {
                    "betti_0_max": int, "betti_0_mean": float,
                    "betti_0_std": float, "betti_0_final": int,
                    ...
                }
            }
        """
        scales = [epsilon_max * i / num_scales for i in range(num_scales + 1)]

        # Compute Betti numbers at each scale
        vectors: dict[int, list[int]] = {
            d: [] for d in range(max_dimension + 1)
        }
        for s in scales:
            betti = betti_numbers(intervals, at_scale=s)
            for d in range(max_dimension + 1):
                vectors[d].append(betti.get(d, 0))

        # Build result
        result: dict = {"scales": scales}
        summary: dict = {}

        for d in range(max_dimension + 1):
            key = f"betti_{d}"
            vec = vectors[d]
            result[key] = vec

            summary[f"{key}_max"] = max(vec) if vec else 0
            summary[f"{key}_mean"] = _mean(vec)
            summary[f"{key}_std"] = _std(vec)
            summary[f"{key}_final"] = vec[-1] if vec else 0

        result["summary"] = summary
        return result


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
