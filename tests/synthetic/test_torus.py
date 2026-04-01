"""Validate persistent homology on T² (torus).

Known topology: β₀=1, β₁=2 (two independent loops), β₂=1 (one void).

Note: Detecting the full torus topology (especially β₂) requires dense
sampling and high max_dimension. This test validates the more robust
β₀ and β₁ signals with a moderate point cloud.
"""

import math

from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.graph import PointCloudGraph
from pn_tda.core.persistence import PersistentHomology, betti_numbers


def _sample_torus(
    n_major: int = 12,
    n_minor: int = 8,
    R: float = 3.0,
    r: float = 1.0,
) -> list[list[float]]:
    """Sample points from a torus in R³.

    R = major radius (center of tube to center of torus)
    r = minor radius (radius of tube)
    """
    points = []
    for i in range(n_major):
        theta = 2 * math.pi * i / n_major
        for j in range(n_minor):
            phi = 2 * math.pi * j / n_minor
            x = (R + r * math.cos(phi)) * math.cos(theta)
            y = (R + r * math.cos(phi)) * math.sin(theta)
            z = r * math.sin(phi)
            points.append([x, y, z])
    return points


def test_torus_connected():
    """Torus should be connected (β₀=1) at moderate scale."""
    points = _sample_torus(n_major=10, n_minor=6, R=3.0, r=1.0)
    graph = PointCloudGraph(points)

    # Use max_dimension=1 for faster computation; enough to check β₀
    builder = VietorisRipsBuilder(epsilon_max=3.0, max_dimension=1)
    st = builder.build(graph)

    ph = PersistentHomology()
    intervals = ph.compute(st)

    betti = betti_numbers(intervals, at_scale=2.5)
    assert betti.get(0, 0) == 1, f"Expected β₀=1, got {betti.get(0, 0)}"


def test_torus_has_two_h1_loops():
    """Torus should have β₁ >= 2 (two independent loops) at appropriate scale."""
    points = _sample_torus(n_major=10, n_minor=6, R=3.0, r=1.0)
    graph = PointCloudGraph(points)

    builder = VietorisRipsBuilder(epsilon_max=3.5, max_dimension=2)
    st = builder.build(graph)

    ph = PersistentHomology()
    intervals = ph.compute(st)

    # Check that we see at least 2 H1 intervals with significant persistence
    h1 = [iv for iv in intervals if iv.dimension == 1 and iv.death != float("inf")]
    significant_h1 = [iv for iv in h1 if iv.persistence > 0.3]
    assert len(significant_h1) >= 2, (
        f"Expected at least 2 significant H1 loops for torus, "
        f"got {len(significant_h1)}: {significant_h1}"
    )
