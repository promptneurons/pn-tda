"""Thread-level maturity scoring via topological features."""

from __future__ import annotations

import math

from pn_tda.core.graph import Graph
from pn_tda.core.persistence import PersistenceInterval, betti_numbers


class ThreadMaturityScorer:
    """Compute maturity scores for GLN threads.

    Components (each 0-1, higher = more mature):
    - connectedness: fraction of nodes in a single component
    - topological_stability: low variance in β₀ across scales
    - persistence_plateau: total persistence slope near 0 at end
    - dimensional_shift: presence of β₁ (structure beyond components)

    Combined into a single maturity_score via equal weighting.
    """

    def score(
        self,
        intervals: list[PersistenceInterval],
        graph: Graph,
        num_scales: int = 10,
        epsilon_max: float = 1.0,
    ) -> dict:
        """Compute maturity score and components.

        Returns:
            {
                "maturity_score": float,  # 0-1
                "connectedness": float,
                "topological_stability": float,
                "persistence_plateau": float,
                "dimensional_shift": float,
            }
        """
        n = len(list(graph.nodes()))
        scales = [epsilon_max * i / num_scales for i in range(num_scales + 1)]

        # Betti numbers at each scale
        b0_values = []
        b1_values = []
        for s in scales:
            betti = betti_numbers(intervals, at_scale=s)
            b0_values.append(betti.get(0, 0))
            b1_values.append(betti.get(1, 0))

        connectedness = self._connectedness(b0_values, n)
        stability = self._topological_stability(b0_values)
        plateau = self._persistence_plateau(intervals, scales)
        dim_shift = self._dimensional_shift(b1_values)

        components = [connectedness, stability, plateau, dim_shift]
        maturity_score = sum(components) / len(components)

        return {
            "maturity_score": maturity_score,
            "connectedness": connectedness,
            "topological_stability": stability,
            "persistence_plateau": plateau,
            "dimensional_shift": dim_shift,
        }

    def _connectedness(self, b0_values: list[int], n: int) -> float:
        """1 - (β₀_final - 1) / (n - 1). Fully connected = 1.0."""
        if n <= 1:
            return 1.0
        # Use the last non-zero β₀, or the final value
        b0_final = b0_values[-1] if b0_values else n
        if b0_final <= 0:
            b0_final = 1
        return max(0.0, 1.0 - (b0_final - 1) / (n - 1))

    def _topological_stability(self, b0_values: list[int]) -> float:
        """1 - normalized variance of β₀ across scales. Low variance = stable."""
        if len(b0_values) < 2:
            return 1.0
        mean = sum(b0_values) / len(b0_values)
        if mean == 0:
            return 1.0
        variance = sum((v - mean) ** 2 for v in b0_values) / len(b0_values)
        # Normalize by mean² to get coefficient of variation squared
        cv_squared = variance / (mean ** 2)
        # Map to 0-1: high cv → low stability
        return max(0.0, 1.0 - math.sqrt(cv_squared))

    def _persistence_plateau(
        self, intervals: list[PersistenceInterval], scales: list[float]
    ) -> float:
        """Check if total persistence has plateaued (slope ≈ 0 at end).

        Uses the last 3 scales to estimate slope of cumulative persistence.
        """
        if len(scales) < 3:
            return 0.5  # Not enough data

        # Compute total alive persistence at each of last 3 scales
        tail_scales = scales[-3:]
        tail_persistence = []
        for s in tail_scales:
            total_p = sum(
                min(iv.death, s) - iv.birth
                for iv in intervals
                if iv.birth <= s < iv.death
            )
            tail_persistence.append(total_p)

        if all(p == 0 for p in tail_persistence):
            return 1.0  # No persistence = trivially plateaued

        # Compute normalized slope
        max_p = max(tail_persistence) if max(tail_persistence) > 0 else 1.0
        normalized = [p / max_p for p in tail_persistence]
        # Slope between first and last of the 3 points
        slope = abs(normalized[-1] - normalized[0])
        # Low slope → high plateau score
        return max(0.0, 1.0 - slope)

    def _dimensional_shift(self, b1_values: list[int]) -> float:
        """1.0 if any β₁ > 0 at any scale (structure beyond components)."""
        if any(b > 0 for b in b1_values):
            return 1.0
        return 0.0
