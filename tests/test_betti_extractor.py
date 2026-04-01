"""Tests for BettiNumberExtractor."""

import math

from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.graph import PointCloudGraph
from pn_tda.core.persistence import PersistenceInterval, PersistentHomology
from pn_tda.features.betti import BettiNumberExtractor


def _circle_intervals():
    """Generate intervals from a circle point cloud."""
    points = [
        [math.cos(2 * math.pi * i / 20), math.sin(2 * math.pi * i / 20)]
        for i in range(20)
    ]
    graph = PointCloudGraph(points)
    builder = VietorisRipsBuilder(epsilon_max=2.0, max_dimension=2)
    st = builder.build(graph)
    return PersistentHomology().compute(st)


def test_extract_shape():
    """Output should have scales, betti vectors, and summary."""
    intervals = _circle_intervals()
    ext = BettiNumberExtractor()
    result = ext.extract(intervals, num_scales=10, epsilon_max=2.0)

    assert "scales" in result
    assert "betti_0" in result
    assert "betti_1" in result
    assert "summary" in result
    assert len(result["scales"]) == 11  # 0..10 inclusive
    assert len(result["betti_0"]) == 11
    assert len(result["betti_1"]) == 11


def test_circle_betti_vectors():
    """Circle should show β₀=1, β₁=1 at middle scales."""
    intervals = _circle_intervals()
    ext = BettiNumberExtractor()
    result = ext.extract(intervals, num_scales=10, epsilon_max=2.0)

    # At scale 1.0 (index 5), should see β₀=1, β₁=1
    assert result["betti_0"][5] == 1
    assert result["betti_1"][5] == 1


def test_summary_stats():
    """Summary should have max, mean, std, final for each dimension."""
    intervals = _circle_intervals()
    ext = BettiNumberExtractor()
    result = ext.extract(intervals, num_scales=10, epsilon_max=2.0)

    summary = result["summary"]
    assert "betti_0_max" in summary
    assert "betti_0_mean" in summary
    assert "betti_0_std" in summary
    assert "betti_0_final" in summary

    # β₀ starts at 20 (20 points) and decreases to 1
    assert summary["betti_0_max"] == 20
    assert summary["betti_0_final"] == 1
    assert summary["betti_0_mean"] > 0
    assert summary["betti_0_std"] > 0


def test_empty_intervals():
    """Empty intervals should return zero vectors."""
    ext = BettiNumberExtractor()
    result = ext.extract([], num_scales=5, epsilon_max=1.0)

    assert all(v == 0 for v in result["betti_0"])
    assert all(v == 0 for v in result["betti_1"])
    assert result["summary"]["betti_0_max"] == 0
    assert result["summary"]["betti_0_mean"] == 0.0


def test_custom_max_dimension():
    """Should include betti_2 when max_dimension=2."""
    intervals = _circle_intervals()
    ext = BettiNumberExtractor()
    result = ext.extract(intervals, num_scales=5, epsilon_max=2.0, max_dimension=2)

    assert "betti_2" in result
    assert "betti_2_max" in result["summary"]
