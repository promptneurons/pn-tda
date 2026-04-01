"""Tests for ThreadMaturityScorer."""

import math

from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.graph import PointCloudGraph
from pn_tda.core.persistence import PersistentHomology
from pn_tda.features.maturity import ThreadMaturityScorer


def _make_pipeline(points, epsilon_max=2.0, max_dimension=2):
    """Helper: point cloud → graph, intervals."""
    graph = PointCloudGraph(points)
    builder = VietorisRipsBuilder(epsilon_max=epsilon_max, max_dimension=max_dimension)
    st = builder.build(graph)
    intervals = PersistentHomology().compute(st)
    return graph, intervals


def test_connected_graph_high_connectedness():
    """Tightly clustered points → connectedness ≈ 1.0."""
    # 5 points very close together → all connect at small epsilon
    points = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05]]
    graph, intervals = _make_pipeline(points, epsilon_max=1.0)

    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=1.0)

    assert result["connectedness"] > 0.9


def test_disconnected_graph_low_connectedness():
    """Two well-separated clusters → connectedness < 1.0 at moderate scale."""
    points = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]]
    graph, intervals = _make_pipeline(points, epsilon_max=1.0, max_dimension=1)

    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=1.0)

    # Two components at final scale → connectedness = 1 - (2-1)/(4-1) = 0.667
    assert result["connectedness"] < 0.9


def test_circle_has_dimensional_shift():
    """Circle has β₁ > 0 → dimensional_shift = 1.0."""
    points = [
        [math.cos(2 * math.pi * i / 20), math.sin(2 * math.pi * i / 20)]
        for i in range(20)
    ]
    graph, intervals = _make_pipeline(points, epsilon_max=2.0)

    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=2.0)

    assert result["dimensional_shift"] == 1.0


def test_no_loops_no_dimensional_shift():
    """Collinear points have no persistent loops → dimensional_shift = 0.0."""
    # Points on a line with max_dimension=2 so triangles fill in and kill H1
    points = [[float(i), 0.0] for i in range(5)]
    graph, intervals = _make_pipeline(points, epsilon_max=2.0, max_dimension=2)

    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=2.0)

    assert result["dimensional_shift"] == 0.0


def test_maturity_score_range():
    """Maturity score should always be in [0, 1]."""
    points = [
        [math.cos(2 * math.pi * i / 10), math.sin(2 * math.pi * i / 10)]
        for i in range(10)
    ]
    graph, intervals = _make_pipeline(points, epsilon_max=2.5)

    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=2.5)

    assert 0.0 <= result["maturity_score"] <= 1.0
    assert 0.0 <= result["connectedness"] <= 1.0
    assert 0.0 <= result["topological_stability"] <= 1.0
    assert 0.0 <= result["persistence_plateau"] <= 1.0
    assert result["dimensional_shift"] in (0.0, 1.0)


def test_all_components_present():
    """Result dict should have all expected keys."""
    points = [[0.0, 0.0], [1.0, 0.0]]
    graph, intervals = _make_pipeline(points, epsilon_max=2.0, max_dimension=1)

    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=2.0)

    assert "maturity_score" in result
    assert "connectedness" in result
    assert "topological_stability" in result
    assert "persistence_plateau" in result
    assert "dimensional_shift" in result
