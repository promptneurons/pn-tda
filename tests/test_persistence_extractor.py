"""Tests for PersistenceFeatureExtractor."""

import math

from pn_tda.core.persistence import PersistenceInterval
from pn_tda.features.persistence import PersistenceFeatureExtractor


def test_known_intervals():
    """Verify features from hand-crafted intervals."""
    intervals = [
        PersistenceInterval(dimension=0, birth=0.0, death=0.5),   # persistence=0.5
        PersistenceInterval(dimension=0, birth=0.0, death=1.0),   # persistence=1.0
        PersistenceInterval(dimension=0, birth=0.0, death=float("inf")),
        PersistenceInterval(dimension=1, birth=0.3, death=0.8),   # persistence=0.5
    ]
    ext = PersistenceFeatureExtractor()
    result = ext.extract(intervals, max_dimension=1)

    # Dim 0: 2 finite intervals
    assert result["dim_0_count"] == 2
    assert result["dim_0_total_persistence"] == 1.5
    assert result["dim_0_mean_persistence"] == 0.75
    assert result["dim_0_max_persistence"] == 1.0

    # Dim 1: 1 finite interval
    assert result["dim_1_count"] == 1
    assert result["dim_1_total_persistence"] == 0.5
    assert result["dim_1_mean_persistence"] == 0.5
    assert result["dim_1_max_persistence"] == 0.5

    # Global
    assert result["total_intervals"] == 4
    assert result["infinite_intervals"] == 1


def test_entropy_single_interval():
    """Single interval should have entropy 0."""
    intervals = [
        PersistenceInterval(dimension=0, birth=0.0, death=1.0),
    ]
    ext = PersistenceFeatureExtractor()
    result = ext.extract(intervals, max_dimension=0)
    assert result["dim_0_entropy"] == 0.0


def test_entropy_equal_intervals():
    """Equal-persistence intervals should have max entropy."""
    intervals = [
        PersistenceInterval(dimension=0, birth=0.0, death=1.0),
        PersistenceInterval(dimension=0, birth=0.0, death=1.0),
        PersistenceInterval(dimension=0, birth=0.0, death=1.0),
    ]
    ext = PersistenceFeatureExtractor()
    result = ext.extract(intervals, max_dimension=0)
    # Entropy of uniform distribution over 3 items = log(3)
    assert abs(result["dim_0_entropy"] - math.log(3)) < 1e-9


def test_no_finite_intervals():
    """Dimension with no finite intervals should have all zeros."""
    intervals = [
        PersistenceInterval(dimension=0, birth=0.0, death=float("inf")),
    ]
    ext = PersistenceFeatureExtractor()
    result = ext.extract(intervals, max_dimension=1)

    assert result["dim_0_count"] == 0
    assert result["dim_0_total_persistence"] == 0.0
    assert result["dim_1_count"] == 0
    assert result["dim_1_total_persistence"] == 0.0
    assert result["infinite_intervals"] == 1


def test_empty_intervals():
    """Empty list should produce all zeros."""
    ext = PersistenceFeatureExtractor()
    result = ext.extract([], max_dimension=1)

    assert result["dim_0_count"] == 0
    assert result["dim_1_count"] == 0
    assert result["total_intervals"] == 0
    assert result["infinite_intervals"] == 0
