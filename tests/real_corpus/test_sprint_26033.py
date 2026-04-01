"""Validation tests on sprint 26033 corpus fixture.

Tests: full pipeline, reproducibility, sanity checks, feature ranges.
"""

import math

from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.persistence import PersistentHomology, betti_numbers
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor


def _run_pipeline(db_path, epsilon_max=1.0, max_dimension=2):
    graph = SignalDBGraph(db_path)
    builder = VietorisRipsBuilder(epsilon_max=epsilon_max, max_dimension=max_dimension)
    st = builder.build(graph)
    intervals = PersistentHomology().compute(st)
    return graph, intervals


def test_pipeline_runs(sprint_26033_db):
    """Full pipeline should complete without errors."""
    graph, intervals = _run_pipeline(sprint_26033_db)
    assert len(intervals) > 0


def test_node_count(sprint_26033_db):
    """Should have 10 documents."""
    graph = SignalDBGraph(sprint_26033_db)
    assert len(list(graph.nodes())) == 10


def test_reproducibility(sprint_26033_db):
    """Running pipeline twice should produce identical results."""
    _, intervals_a = _run_pipeline(sprint_26033_db)
    _, intervals_b = _run_pipeline(sprint_26033_db)

    assert len(intervals_a) == len(intervals_b)
    for a, b in zip(intervals_a, intervals_b):
        assert a.dimension == b.dimension
        assert a.birth == b.birth
        assert a.death == b.death


def test_betti_at_zero_scale(sprint_26033_db):
    """At scale 0, β₀ should equal number of nodes (all disconnected)."""
    graph, intervals = _run_pipeline(sprint_26033_db)
    n = len(list(graph.nodes()))
    betti = betti_numbers(intervals, at_scale=0.0)
    assert betti.get(0, 0) == n


def test_betti_at_large_scale(sprint_26033_db):
    """At large scale, β₀ should be small (most nodes connected)."""
    graph, intervals = _run_pipeline(sprint_26033_db)
    betti = betti_numbers(intervals, at_scale=0.99)
    # With 3 clusters + 2 isolated, expect β₀ between 1 and 5
    assert 1 <= betti.get(0, 0) <= 5


def test_persistence_features_non_negative(sprint_26033_db):
    """All persistence features should be >= 0."""
    _, intervals = _run_pipeline(sprint_26033_db)
    ext = PersistenceFeatureExtractor()
    features = ext.extract(intervals, max_dimension=2)

    for key, val in features.items():
        assert val >= 0, f"{key} = {val} is negative"


def test_betti_vector_shape(sprint_26033_db):
    """Betti vectors should have correct number of scales."""
    _, intervals = _run_pipeline(sprint_26033_db)
    ext = BettiNumberExtractor()
    result = ext.extract(intervals, num_scales=10, epsilon_max=1.0)

    assert len(result["scales"]) == 11
    assert len(result["betti_0"]) == 11
    assert len(result["betti_1"]) == 11
    assert len(result["betti_2"]) == 11


def test_maturity_score_range(sprint_26033_db):
    """Maturity score and components should be in [0, 1]."""
    graph, intervals = _run_pipeline(sprint_26033_db)
    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=1.0)

    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"


def test_has_multiple_components(sprint_26033_db):
    """Sprint 26033 has isolated docs, so should show multiple components."""
    graph, intervals = _run_pipeline(sprint_26033_db)
    # At small scale, should have more than 1 component
    betti = betti_numbers(intervals, at_scale=0.01)
    assert betti.get(0, 0) > 1
