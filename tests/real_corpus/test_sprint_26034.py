"""Validation tests on sprint 26034 corpus fixture.

Tests: pipeline validity, cross-sprint comparison, temporal consistency.
"""

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


def test_pipeline_runs(sprint_26034_db):
    """Full pipeline should complete without errors."""
    graph, intervals = _run_pipeline(sprint_26034_db)
    assert len(intervals) > 0


def test_node_count(sprint_26034_db):
    """Should have 8 documents."""
    graph = SignalDBGraph(sprint_26034_db)
    assert len(list(graph.nodes())) == 8


def test_different_from_26033(sprint_26033_db, sprint_26034_db):
    """Different sprints should produce different topological features."""
    _, intervals_33 = _run_pipeline(sprint_26033_db)
    _, intervals_34 = _run_pipeline(sprint_26034_db)

    ext = PersistenceFeatureExtractor()
    features_33 = ext.extract(intervals_33)
    features_34 = ext.extract(intervals_34)

    # At least one feature should differ
    differences = sum(
        1 for k in features_33
        if features_33[k] != features_34[k]
    )
    assert differences > 0, "Sprints 26033 and 26034 produced identical features"


def test_26034_more_connected(sprint_26033_db, sprint_26034_db):
    """Sprint 26034 has denser edges → should have higher connectedness."""
    graph_33, intervals_33 = _run_pipeline(sprint_26033_db)
    graph_34, intervals_34 = _run_pipeline(sprint_26034_db)

    scorer = ThreadMaturityScorer()
    maturity_33 = scorer.score(intervals_33, graph_33, epsilon_max=1.0)
    maturity_34 = scorer.score(intervals_34, graph_34, epsilon_max=1.0)

    # 26034 is designed with more edges → higher connectedness
    assert maturity_34["connectedness"] >= maturity_33["connectedness"]


def test_betti_vector_shape(sprint_26034_db):
    """Betti vectors should have correct shape."""
    _, intervals = _run_pipeline(sprint_26034_db)
    ext = BettiNumberExtractor()
    result = ext.extract(intervals, num_scales=10, epsilon_max=1.0)

    assert len(result["scales"]) == 11
    assert len(result["betti_0"]) == 11


def test_maturity_score_range(sprint_26034_db):
    """All maturity components should be in [0, 1]."""
    graph, intervals = _run_pipeline(sprint_26034_db)
    scorer = ThreadMaturityScorer()
    result = scorer.score(intervals, graph, epsilon_max=1.0)

    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"


def test_reproducibility(sprint_26034_db):
    """Running pipeline twice should produce identical results."""
    _, intervals_a = _run_pipeline(sprint_26034_db)
    _, intervals_b = _run_pipeline(sprint_26034_db)

    assert len(intervals_a) == len(intervals_b)
    for a, b in zip(intervals_a, intervals_b):
        assert a.dimension == b.dimension
        assert a.birth == b.birth
        assert a.death == b.death
