"""Distance metrics for graph construction."""

import math


def jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard distance: 1 - |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
