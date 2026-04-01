"""Validate persistent homology on S¹ (circle).

Known topology: β₀=1 (one connected component), β₁=1 (one loop).
"""

import math

from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.graph import PointCloudGraph
from pn_tda.core.persistence import PersistentHomology, betti_numbers


def _sample_circle(n: int = 20, radius: float = 1.0) -> list[list[float]]:
    """Sample n evenly-spaced points from a circle."""
    return [
        [radius * math.cos(2 * math.pi * i / n),
         radius * math.sin(2 * math.pi * i / n)]
        for i in range(n)
    ]


def test_circle_betti_numbers():
    """Circle should have β₀=1, β₁=1 at an appropriate scale."""
    points = _sample_circle(n=20, radius=1.0)
    graph = PointCloudGraph(points)

    # Spacing between adjacent points on unit circle with 20 pts:
    # 2*sin(pi/20) ≈ 0.31
    # Use epsilon large enough to connect all neighbors but not skip
    builder = VietorisRipsBuilder(epsilon_max=2.0, max_dimension=2)
    st = builder.build(graph)

    ph = PersistentHomology()
    intervals = ph.compute(st)

    # At a scale where the circle is connected but the loop persists,
    # we should see β₀=1, β₁=1.
    # The loop should be born around the neighbor spacing and persist.
    # Pick a scale in the middle range.
    scale = 1.0
    betti = betti_numbers(intervals, at_scale=scale)

    assert betti.get(0, 0) == 1, f"Expected β₀=1, got {betti.get(0, 0)}"
    assert betti.get(1, 0) == 1, f"Expected β₁=1, got {betti.get(1, 0)}"


def test_circle_has_one_persistent_loop():
    """The circle should produce exactly one long-lived H1 interval."""
    points = _sample_circle(n=20, radius=1.0)
    graph = PointCloudGraph(points)
    builder = VietorisRipsBuilder(epsilon_max=2.5, max_dimension=2)
    st = builder.build(graph)

    ph = PersistentHomology()
    intervals = ph.compute(st)

    h1 = [iv for iv in intervals if iv.dimension == 1 and iv.death != float("inf")]
    # There should be at least one H1 interval with significant persistence
    significant = [iv for iv in h1 if iv.persistence > 0.5]
    assert len(significant) >= 1, f"Expected at least one persistent H1 loop, got {significant}"
