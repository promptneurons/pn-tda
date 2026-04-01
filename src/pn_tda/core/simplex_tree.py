"""Simplex Tree data structure for efficient simplicial complex storage."""

from __future__ import annotations

from typing import Optional


class SimplexTreeNode:
    """A node in the simplex tree."""

    __slots__ = ("vertex", "filtration_value", "children", "parent")

    def __init__(
        self,
        vertex: int,
        filtration_value: float = 0.0,
        parent: Optional["SimplexTreeNode"] = None,
    ):
        self.vertex = vertex
        self.filtration_value = filtration_value
        self.children: dict[int, SimplexTreeNode] = {}
        self.parent = parent


class SimplexTree:
    """Simplex Tree for storing a filtered simplicial complex.

    Supports insertion, lookup, and sorted enumeration of simplices
    by filtration value. Based on Boissonnat et al. (2016).
    """

    def __init__(self):
        self._root = SimplexTreeNode(vertex=-1, filtration_value=-1.0)
        self._num_simplices = 0
        self._max_dimension = -1

    def insert(self, simplex: tuple[int, ...], filtration_value: float = 0.0) -> None:
        """Insert a simplex with the given filtration value.

        Also inserts all faces (sub-simplices) with filtration value
        equal to the minimum of the existing value and the new value.
        """
        simplex = tuple(sorted(simplex))
        # Insert all faces
        for k in range(1, len(simplex) + 1):
            for face in _combinations(simplex, k):
                self._insert_single(face, filtration_value)

    def _insert_single(self, simplex: tuple[int, ...], filtration_value: float) -> None:
        """Insert a single simplex (without auto-inserting faces)."""
        node = self._root
        for v in simplex:
            if v not in node.children:
                child = SimplexTreeNode(vertex=v, filtration_value=filtration_value, parent=node)
                node.children[v] = child
                self._num_simplices += 1
                dim = len(simplex) - 1
                if dim > self._max_dimension:
                    self._max_dimension = dim
            else:
                # Keep the minimum filtration value (earliest birth time).
                # Faces must be born no later than their cofaces.
                if filtration_value < node.children[v].filtration_value:
                    node.children[v].filtration_value = filtration_value
            node = node.children[v]

    def find(self, simplex: tuple[int, ...]) -> bool:
        """Check if a simplex exists in the tree."""
        simplex = tuple(sorted(simplex))
        node = self._root
        for v in simplex:
            if v not in node.children:
                return False
            node = node.children[v]
        return True

    def filtration(self, simplex: tuple[int, ...]) -> float:
        """Return the filtration value of a simplex. Raises KeyError if not found."""
        simplex = tuple(sorted(simplex))
        node = self._root
        for v in simplex:
            if v not in node.children:
                raise KeyError(f"Simplex {simplex} not found")
            node = node.children[v]
        return node.filtration_value

    def get_simplices(self) -> list[tuple[tuple[int, ...], float]]:
        """Return all simplices sorted by (filtration_value, dimension, vertices)."""
        result: list[tuple[tuple[int, ...], float]] = []
        self._collect(self._root, [], result)
        result.sort(key=lambda x: (x[1], len(x[0]), x[0]))
        return result

    def _collect(
        self,
        node: SimplexTreeNode,
        prefix: list[int],
        result: list[tuple[tuple[int, ...], float]],
    ) -> None:
        for v, child in sorted(node.children.items()):
            simplex = prefix + [v]
            result.append((tuple(simplex), child.filtration_value))
            self._collect(child, simplex, result)

    def num_simplices(self) -> int:
        return self._num_simplices

    def dimension(self) -> int:
        return self._max_dimension


def _combinations(seq: tuple[int, ...], k: int) -> list[tuple[int, ...]]:
    """Generate all k-element combinations from seq."""
    if k == 0:
        return [()]
    if k > len(seq):
        return []
    result = []
    for i, elem in enumerate(seq):
        for rest in _combinations(seq[i + 1 :], k - 1):
            result.append((elem,) + rest)
    return result
