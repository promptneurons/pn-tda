"""Filtration construction from graphs.

Two builders:
- VietorisRipsBuilder: O(n²) all-pairs distance, for point clouds
- GraphFiltrationBuilder: O(|V|+|E|) from existing edges, for knowledge graphs
"""

from collections import defaultdict
from itertools import combinations

from pn_tda.core.graph import Graph
from pn_tda.core.simplex_tree import SimplexTree


class GraphFiltrationBuilder:
    """Build a filtered simplicial complex directly from graph edges.

    Instead of computing all-pairs distances (O(n²)), uses the edges
    already present in the graph as the 1-skeleton. Filtration values
    are derived from edge-local signal distance. Clique expansion is
    sparse — only triangles where all 3 edges exist in the graph.

    Complexity: O(|V| + |E| * max_degree) instead of O(n²).
    """

    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension

    def build(self, graph: Graph) -> SimplexTree:
        """Construct a filtered simplicial complex from existing graph edges."""
        st = SimplexTree()
        node_list = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        # Add 0-simplices (vertices) at filtration value 0
        for i in range(len(node_list)):
            st.insert((i,), 0.0)

        # Add 1-simplices from existing edges with edge-local distance
        # Only compute distance for connected pairs — O(|E|) not O(n²)
        adjacency: dict[int, set[int]] = defaultdict(set)
        edge_filt: dict[tuple[int, int], float] = {}

        for src, tgt in graph.edges():
            if src not in node_to_idx or tgt not in node_to_idx:
                continue
            i, j = node_to_idx[src], node_to_idx[tgt]
            i, j = min(i, j), max(i, j)
            if (i, j) in edge_filt:
                continue  # dedupe

            d = graph.get_distance(src, tgt)
            edge_filt[(i, j)] = d
            st.insert((i, j), d)
            adjacency[i].add(j)
            adjacency[j].add(i)

        # Sparse clique expansion for triangles (dim 2)
        # For each edge (i,j), check shared neighbors — O(|E| * max_degree)
        if self.max_dimension >= 2:
            for (i, j), d_ij in edge_filt.items():
                shared = adjacency[i] & adjacency[j]
                for k in shared:
                    if k <= j:
                        continue  # canonical ordering: i < j < k
                    # All three edges exist — form triangle
                    d_ik = edge_filt.get((min(i, k), max(i, k)), 0.0)
                    d_jk = edge_filt.get((min(j, k), max(j, k)), 0.0)
                    tri_filt = max(d_ij, d_ik, d_jk)
                    st._insert_single((i, j, k), tri_filt)

        # Tetrahedra (dim 3) if requested — sparse check on triangles
        if self.max_dimension >= 3:
            # For each triangle (i,j,k), check shared neighbors of all 3
            triangles = [
                (s, f) for s, f in st.get_simplices() if len(s) == 3
            ]
            for (i, j, k), tri_f in triangles:
                shared = adjacency[i] & adjacency[j] & adjacency[k]
                for l in shared:
                    if l <= k:
                        continue
                    d_il = edge_filt.get((min(i, l), max(i, l)), 0.0)
                    d_jl = edge_filt.get((min(j, l), max(j, l)), 0.0)
                    d_kl = edge_filt.get((min(k, l), max(k, l)), 0.0)
                    tet_filt = max(tri_f, d_il, d_jl, d_kl)
                    st._insert_single((i, j, k, l), tet_filt)

        return st


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
