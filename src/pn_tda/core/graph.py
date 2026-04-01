"""Abstract graph interface for TDA computation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple


class Graph(ABC):
    """Abstract interface for graph data structures.

    All TDA adapters implement this interface, allowing algorithms
    to work with any data source (obsidian-refs, SignalDB, Neo4j, etc.).
    """

    @abstractmethod
    def nodes(self) -> Iterable[str]:
        """Return all node IDs."""

    @abstractmethod
    def edges(self) -> Iterable[Tuple[str, str]]:
        """Return all edges as (source, target) tuples."""

    @abstractmethod
    def get_distance(self, u: str, v: str) -> float:
        """Return distance between two nodes.

        For direct graphs: returns 1 - similarity or shortest path length.
        For weighted graphs: returns the weight directly.
        """

    @abstractmethod
    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Return attributes for a node (type, properties, etc.)."""

    @abstractmethod
    def get_neighbors(self, node_id: str) -> Iterable[str]:
        """Return immediate neighbors of a node."""


class PointCloudGraph(Graph):
    """Graph constructed from a point cloud with Euclidean distances.

    Useful for synthetic topology tests (circle, torus, etc.).
    """

    def __init__(self, points: list[list[float]]):
        import numpy as np

        self._points = np.asarray(points, dtype=float)
        self._n = len(points)
        self._node_ids = [str(i) for i in range(self._n)]

        # Pre-compute pairwise distance matrix
        diff = self._points[:, None, :] - self._points[None, :, :]
        self._dist_matrix = np.sqrt((diff**2).sum(axis=-1))

    def nodes(self) -> Iterable[str]:
        return self._node_ids

    def edges(self) -> Iterable[Tuple[str, str]]:
        for i in range(self._n):
            for j in range(i + 1, self._n):
                yield (str(i), str(j))

    def get_distance(self, u: str, v: str) -> float:
        return float(self._dist_matrix[int(u), int(v)])

    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        idx = int(node_id)
        return {"point": self._points[idx].tolist()}

    def get_neighbors(self, node_id: str) -> Iterable[str]:
        idx = int(node_id)
        return [str(j) for j in range(self._n) if j != idx]
