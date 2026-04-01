"""Validate persistent homology on S¹ ⊔ S¹ (two disjoint circles).

Known topology: β₀=2 (two components), β₁=2 (two loops).
"""

import math

from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.graph import PointCloudGraph
from pn_tda.core.persistence import PersistentHomology, betti_numbers


def _sample_two_circles(
    n: int = 16, radius: float = 1.0, separation: float = 10.0
) -> list[list[float]]:
    """Sample points from two well-separated circles."""
    points = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        points.append([radius * math.cos(angle), radius * math.sin(angle)])
    for i in range(n):
        angle = 2 * math.pi * i / n
        points.append([
            separation + radius * math.cos(angle),
            radius * math.sin(angle),
        ])
    return points


def test_two_circles_betti_numbers():
    """Two separated circles should have β₀=2, β₁=2 at appropriate scale."""
    points = _sample_two_circles(n=16, radius=1.0, separation=10.0)
    graph = PointCloudGraph(points)

    # Epsilon large enough to connect each circle but not bridge the gap
    builder = VietorisRipsBuilder(epsilon_max=2.5, max_dimension=2)
    st = builder.build(graph)

    ph = PersistentHomology()
    intervals = ph.compute(st)

    scale = 1.0
    betti = betti_numbers(intervals, at_scale=scale)

    assert betti.get(0, 0) == 2, f"Expected β₀=2, got {betti.get(0, 0)}"
    assert betti.get(1, 0) == 2, f"Expected β₁=2, got {betti.get(1, 0)}"


def test_two_circles_components_merge_at_large_scale():
    """At very large scale, two circles should merge into one component."""
    points = _sample_two_circles(n=16, radius=1.0, separation=5.0)
    graph = PointCloudGraph(points)

    builder = VietorisRipsBuilder(epsilon_max=6.0, max_dimension=1)
    st = builder.build(graph)

    ph = PersistentHomology()
    intervals = ph.compute(st)

    # At scale > separation, everything merges
    betti = betti_numbers(intervals, at_scale=5.5)
    assert betti.get(0, 0) == 1, f"Expected β₀=1 at large scale, got {betti.get(0, 0)}"
