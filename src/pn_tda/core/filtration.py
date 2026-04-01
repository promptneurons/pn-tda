"""Filtration construction from graphs (Vietoris-Rips complex)."""

from itertools import combinations

from pn_tda.core.graph import Graph
from pn_tda.core.simplex_tree import SimplexTree


class VietorisRipsBuilder:
    """Build a Vietoris-Rips complex from a graph.

    The VR complex at scale epsilon contains a simplex [v0, ..., vk]
    if and only if d(vi, vj) <= epsilon for all pairs i, j.
    """

    def __init__(self, epsilon_max: float = 1.0, max_dimension: int = 2):
        self.epsilon_max = epsilon_max
        self.max_dimension = max_dimension

    def build(self, graph: Graph) -> SimplexTree:
        """Construct a filtered simplicial complex from the graph."""
        st = SimplexTree()
        node_list = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)

        # Add 0-simplices (vertices) at filtration value 0
        for i in range(n):
            st.insert((i,), 0.0)

        # Pre-compute pairwise distances and add 1-simplices (edges)
        dist = {}
        for i in range(n):
            for j in range(i + 1, n):
                d = graph.get_distance(node_list[i], node_list[j])
                if d <= self.epsilon_max:
                    dist[(i, j)] = d
                    st.insert((i, j), d)

        # Add higher-dimensional simplices (clique expansion)
        for dim in range(2, self.max_dimension + 1):
            for combo in combinations(range(n), dim + 1):
                # Check if all edges exist within epsilon
                max_dist = 0.0
                all_edges_exist = True
                for a, b in combinations(combo, 2):
                    key = (min(a, b), max(a, b))
                    if key not in dist:
                        all_edges_exist = False
                        break
                    max_dist = max(max_dist, dist[key])

                if all_edges_exist:
                    # Simplex appears at the maximum edge distance
                    st._insert_single(combo, max_dist)

        return st
