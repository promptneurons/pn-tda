"""End-to-end feature extraction pipeline tests.

Validates all 3 extractors produce well-formed output on both sprints.
"""

import math

from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.persistence import PersistentHomology
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor


def _full_features(db_path):
    """Run all extractors and return combined feature dict."""
    graph = SignalDBGraph(db_path)
    builder = VietorisRipsBuilder(epsilon_max=1.0, max_dimension=2)
    st = builder.build(graph)
    intervals = PersistentHomology().compute(st)

    betti_ext = BettiNumberExtractor()
    persist_ext = PersistenceFeatureExtractor()
    maturity_scorer = ThreadMaturityScorer()

    return {
        "betti": betti_ext.extract(intervals, num_scales=10, epsilon_max=1.0),
        "persistence": persist_ext.extract(intervals, max_dimension=2),
        "maturity": maturity_scorer.score(intervals, graph, epsilon_max=1.0),
    }


def test_betti_output_schema_26033(sprint_26033_db):
    """Betti output should match design doc JSON schema."""
    features = _full_features(sprint_26033_db)
    betti = features["betti"]

    assert isinstance(betti["scales"], list)
    assert isinstance(betti["betti_0"], list)
    assert isinstance(betti["betti_1"], list)
    assert isinstance(betti["betti_2"], list)
    assert isinstance(betti["summary"], dict)

    summary = betti["summary"]
    for dim in range(3):
        prefix = f"betti_{dim}"
        assert f"{prefix}_max" in summary
        assert f"{prefix}_mean" in summary
        assert f"{prefix}_std" in summary
        assert f"{prefix}_final" in summary


def test_persistence_output_schema_26033(sprint_26033_db):
    """Persistence output should have all expected keys."""
    features = _full_features(sprint_26033_db)
    p = features["persistence"]

    for dim in range(3):
        prefix = f"dim_{dim}"
        assert f"{prefix}_count" in p
        assert f"{prefix}_total_persistence" in p
        assert f"{prefix}_mean_persistence" in p
        assert f"{prefix}_max_persistence" in p
        assert f"{prefix}_entropy" in p

    assert "total_intervals" in p
    assert "infinite_intervals" in p


def test_maturity_output_schema_26033(sprint_26033_db):
    """Maturity output should have all expected keys."""
    features = _full_features(sprint_26033_db)
    m = features["maturity"]

    assert "maturity_score" in m
    assert "connectedness" in m
    assert "topological_stability" in m
    assert "persistence_plateau" in m
    assert "dimensional_shift" in m


def test_no_nan_in_features_26033(sprint_26033_db):
    """No NaN values in any numeric features."""
    features = _full_features(sprint_26033_db)

    # Check persistence features
    for key, val in features["persistence"].items():
        assert not math.isnan(val), f"persistence.{key} is NaN"

    # Check maturity features
    for key, val in features["maturity"].items():
        assert not math.isnan(val), f"maturity.{key} is NaN"

    # Check betti summary
    for key, val in features["betti"]["summary"].items():
        assert not math.isnan(val), f"betti.summary.{key} is NaN"


def test_no_nan_in_features_26034(sprint_26034_db):
    """No NaN values in any numeric features for sprint 26034."""
    features = _full_features(sprint_26034_db)

    for key, val in features["persistence"].items():
        assert not math.isnan(val), f"persistence.{key} is NaN"

    for key, val in features["maturity"].items():
        assert not math.isnan(val), f"maturity.{key} is NaN"

    for key, val in features["betti"]["summary"].items():
        assert not math.isnan(val), f"betti.summary.{key} is NaN"


def test_betti_vectors_no_negative(sprint_26033_db):
    """Betti numbers should never be negative."""
    features = _full_features(sprint_26033_db)
    for dim in range(3):
        vec = features["betti"][f"betti_{dim}"]
        assert all(v >= 0 for v in vec), f"betti_{dim} has negative values: {vec}"


def test_infinite_intervals_count(sprint_26033_db):
    """Should have at least 1 infinite interval (the final H0 class)."""
    features = _full_features(sprint_26033_db)
    assert features["persistence"]["infinite_intervals"] >= 1


def test_total_intervals_consistent(sprint_26033_db):
    """total_intervals should equal sum of finite counts + infinite count."""
    features = _full_features(sprint_26033_db)
    p = features["persistence"]
    finite_sum = sum(p[f"dim_{d}_count"] for d in range(3))
    assert p["total_intervals"] == finite_sum + p["infinite_intervals"]
