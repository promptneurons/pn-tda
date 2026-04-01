"""Persistent homology computation via boundary matrix reduction."""

from __future__ import annotations

from dataclasses import dataclass

from pn_tda.core.simplex_tree import SimplexTree


@dataclass(frozen=True)
class PersistenceInterval:
    """A single persistence interval (birth, death) in a given dimension."""

    dimension: int
    birth: float
    death: float  # float('inf') for classes that never die

    @property
    def persistence(self) -> float:
        if self.death == float("inf"):
            return float("inf")
        return self.death - self.birth


class PersistentHomology:
    """Compute persistent homology of a filtered simplicial complex.

    Uses the standard boundary matrix reduction algorithm
    (column-wise left-to-right Gaussian elimination).
    """

    def compute(self, simplex_tree: SimplexTree) -> list[PersistenceInterval]:
        """Compute persistence intervals from a simplex tree.

        Returns a list of PersistenceInterval with (dimension, birth, death).
        Infinite intervals have death = float('inf').
        """
        simplices = simplex_tree.get_simplices()
        if not simplices:
            return []

        # Map each simplex to its column index
        simplex_to_idx: dict[tuple[int, ...], int] = {}
        for idx, (simplex, _filt) in enumerate(simplices):
            simplex_to_idx[simplex] = idx

        n = len(simplices)

        # Build boundary matrix as sparse columns (sets of row indices)
        # boundary[j] = set of row indices where ∂[j] has a 1
        boundary: list[set[int]] = []
        for simplex, _filt in simplices:
            if len(simplex) <= 1:
                # 0-simplices have empty boundary
                boundary.append(set())
            else:
                # k-simplex boundary = sum of (k-1)-faces
                faces = set()
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1 :]
                    if face in simplex_to_idx:
                        faces.add(simplex_to_idx[face])
                boundary.append(faces)

        # Column reduction (standard algorithm over Z/2Z)
        low: dict[int, int] = {}  # low[j] = lowest row index in column j
        pivot_col: dict[int, int] = {}  # pivot_col[row] = column with that lowest row

        for j in range(n):
            while boundary[j]:
                low_j = max(boundary[j])
                if low_j in pivot_col:
                    # Add the pivot column to column j (XOR over Z/2Z)
                    other = pivot_col[low_j]
                    boundary[j] = boundary[j].symmetric_difference(boundary[other])
                else:
                    break

            if boundary[j]:
                low_j = max(boundary[j])
                low[j] = low_j
                pivot_col[low_j] = j

        # Extract persistence intervals
        intervals: list[PersistenceInterval] = []
        paired: set[int] = set()

        for j in range(n):
            if j in low:
                # Column j is a "death" — paired with its low row
                creator_idx = low[j]
                paired.add(creator_idx)
                paired.add(j)

                creator_simplex, birth_val = simplices[creator_idx]
                _destroyer_simplex, death_val = simplices[j]
                dim = len(creator_simplex) - 1

                # Skip zero-persistence intervals
                if death_val > birth_val:
                    intervals.append(
                        PersistenceInterval(
                            dimension=dim, birth=birth_val, death=death_val
                        )
                    )

        # Unpaired simplices give infinite intervals
        for idx in range(n):
            if idx not in paired:
                simplex, birth_val = simplices[idx]
                dim = len(simplex) - 1
                intervals.append(
                    PersistenceInterval(
                        dimension=dim, birth=birth_val, death=float("inf")
                    )
                )

        intervals.sort(key=lambda iv: (iv.dimension, iv.birth, iv.death))
        return intervals


def betti_numbers(
    intervals: list[PersistenceInterval], at_scale: float
) -> dict[int, int]:
    """Compute Betti numbers at a given filtration scale.

    β_k(ε) = number of intervals in dimension k that are alive at scale ε,
    i.e., birth <= ε < death.
    """
    betti: dict[int, int] = {}
    for iv in intervals:
        if iv.birth <= at_scale < iv.death:
            betti[iv.dimension] = betti.get(iv.dimension, 0) + 1
    return betti
