# TDA Library Design: Topological Data Analysis for Knowledge Graphs

**Date:** 2026-03-31
**Status:** Design Approved - Ready for Implementation
**Scope:** Part 1 - General TDA Library for pn-monorepo
**Next Phase:** Part 2 - CASS/cm Integration

---

## Executive Summary

This document presents the design for a Topological Data Analysis (TDA) library to extract topological features from knowledge graphs. The library will:

1. **Compute persistent homology** on document corpora using Vietoris-Rips complexes
2. **Extract topological features**: Betti numbers, persistence diagrams, thread maturity scores
3. **Integrate with pn-autoresearch** as a new scorer for search quality optimization
4. **Validate streaming approach** with incremental Simplex Tree updates
5. **Optionally optimize hot path** with Rust core + Python bindings

The design follows Grok's proposal (Grok Answer, 290256TMAR26) and NotebookLM's recommendation (290410TMAR26), focusing on reusable infrastructure that can later be applied to CASS memory systems.

---

## Table of Contents

1. [Architecture and Project Structure](#1-architecture-and-project-structure)
2. [Component Details](#2-component-details)
3. [Feature Extraction and Integration](#3-feature-extraction-and-integration)
4. [Testing and Validation](#4-testing-and-validation)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Risks and Alternatives](#6-risks-and-alternatives)
7. [Documentation and Handoff](#7-documentation-and-handoff)
8. [Appendix: Clarifying Questions](#appendix-clarifying-questions)

---

## 1. Architecture and Project Structure

### 1.1 Project Layout

```
pn-tda/
├── Cargo.toml                           # Rust crate (production phase)
├── pyproject.toml                       # Python package (prototype phase)
├── README.md
├── docs/
│   ├── DESIGN.md                        # This document
│   ├── API.md                           # API reference
│   ├── INSTALL.md                       # Installation instructions
│   ├── TUTORIAL.md                      # Getting started tutorial
│   ├── RESEARCH.md                      # Findings from prototype phase
│   ├── ALTERNATIVES.md                  # Alternative approaches explored
│   ├── BENCHMARKS.md                    # Performance benchmarks (Phase 3)
│   └── MIGRATION.md                     # Python → Rust migration guide (Phase 4)
├── python/
│   ├── pn_tda/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py                 # Graph abstraction
│   │   │   ├── filtration.py            # Filtration construction
│   │   │   ├── persistence.py           # Persistent homology algorithms
│   │   │   └── simplex_tree.py          # Simplex Tree data structure
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # Graph interface
│   │   │   ├── obsidian_refs.py         # obsidian-refs.json adapter
│   │   │   ├── signal_db.py             # SignalDB/SQLite adapter
│   │   │   └── neo4j_csv.py             # Neo4j CSV import adapter
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── betti.py                 # Betti number extraction
│   │   │   ├── persistence.py           # Persistence diagram/barcode
│   │   │   ├── vectors.py               # Feature vector generation
│   │   │   └── maturity.py              # Thread-level maturity scoring
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── geometry.py              # Distance metrics
│   │       └── visualization.py         # Plotting (optional)
├── rust/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── simplex_tree.rs             # Optimized Simplex Tree
│   │   ├── persistence.rs              # Matrix reduction for PH
│   │   └── python.rs                   # PyO3 bindings
│   └── Cargo.toml
├── tests/
│   ├── synthetic/
│   │   ├── test_circle.py              # Known topology: S¹ (β₀=1, β₁=1)
│   │   ├── test_torus.py               # Known topology: T² (β₀=1, β₁=2, β₂=1)
│   │   └── test_two_circles.py         # Known topology: disconnected S¹ ⊔ S¹
│   └── real_corpus/
│       ├── fixtures/                   # Sample 26033 data
│       ├── test_sprint_26033.py
│       └── test_sprint_26034.py
└── examples/
    └── pn_autoresearch_integration/
        ├── tda_scorer.py               # pn-autoresearch scorer plugin
        └── experiment_tda.yml          # Example config
```

### 1.2 Architecture Layers

**Layer 1: Data Ingestion (Adapters)**
- Implements `Graph` interface: `nodes()`, `edges()`, `get_distance(u, v)`
- Adapters: `ObsidianRefsGraph`, `SignalDBGraph`, `Neo4jCSVGraph`
- Output: Standardized graph representation

**Layer 2: Filtration Construction**
- Builds Vietoris-Rips complex from distance matrix
- Implements incremental filtration: `add_simplex(simplex, filtration_value)`
- Maintains birth/death tracking for persistence

**Layer 3: Persistent Homology Computation**
- Simplex Tree data structure for efficient complex storage
- Matrix reduction algorithm for computing birth/death times
- Output: Persistence diagram {(birth, death, dimension)}

**Layer 4: Feature Extraction**
- Betti numbers per filtration level
- Persistence statistics (total persistence, entropy, longest bars)
- Thread-level maturity scores (GLN path aggregation)

**Layer 5: Integration**
- pn-autoresearch scorer plugin
- Configurable via experiment.yml
- Caches features in SignalDB for fast retrieval

### 1.3 Data Flow

```
obsidian-refs.json
        ↓
  [ObsidianRefsGraph]
        ↓
   Graph Interface
        ↓
  [FiltrationBuilder] → Vietoris-Rips complex
        ↓
  [SimplexTree] → Incremental insertion
        ↓
  [PersistenceComputation] → Matrix reduction
        ↓
  Persistence Diagram {(birth, death, dim)}
        ↓
  [FeatureExtractor] → Betti, vectors, maturity
        ↓
  [TDAScorer] → pn-autoresearch fusion
        ↓
    NDCG@10
```

---

## 2. Component Details

### 2.1 Graph Abstraction Layer

**Interface Definition (Python prototype):**

```python
# python/pn_tda/core/graph.py
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict, Any

class Graph(ABC):
    """Abstract interface for graph data structures."""

    @abstractmethod
    def nodes(self) -> Iterable[str]:
        """Return all node IDs."""
        pass

    @abstractmethod
    def edges(self) -> Iterable[Tuple[str, str]]:
        """Return all edges as (source, target) tuples."""
        pass

    @abstractmethod
    def get_distance(self, u: str, v: str) -> float:
        """Return distance between two nodes.

        For direct graphs: returns 1 - similarity or shortest path length.
        For weighted graphs: returns the weight directly.
        """
        pass

    @abstractmethod
    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Return attributes for a node (type, properties, etc.)."""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str) -> Iterable[str]:
        """Return immediate neighbors of a node."""
        pass
```

**Why this interface:**
- Clean separation between TDA algorithms and data sources
- Each adapter implements the same interface, swap data sources without changing algorithms
- Methods map directly to what TDA algorithms need (no extraneous operations)

### 2.2 Data Adapters

#### ObsidianRefsGraph Adapter

Loads graph from Open Planter `obsidian-refs.json` output. Uses Jaccard distance on neighbor sets to capture semantic similarity through co-occurrence.

#### SignalDBGraph Adapter

Loads graph from azure-wiki-analysis SQLite output. Reuses existing signal extraction infrastructure. Computes distance based on signal co-occurrence between documents.

#### Neo4jCSVGraph Adapter

Loads graph from Neo4j CSV export files. Useful for integrating with existing Neo4j-based knowledge graphs.

### 2.3 Filtration Construction

**VietorisRipsBuilder:**
- Pre-computes distance matrix (O(n²))
- Adds 0-simplices (nodes) at filtration value 0
- Adds 1-simplices (edges) at their distance threshold
- Adds higher-dimensional simplices as cliques (triangle exists iff all 3 edges exist)
- Sorts all simplices by filtration value for efficient persistence computation

**Key parameters:**
- `epsilon_max`: Maximum distance threshold (filters noisy long-distance connections)
- `max_dimension`: Maximum simplex dimension to compute (0=nodes, 1=edges, 2=triangles)

### 2.4 Persistent Homology Computation

**SimplexTreeNode:**
- Represents a single simplex in the tree
- Stores vertex, filtration value, children, parent
- Enables efficient traversal and coface lookup

**SimplexTree:**
- Dynamic data structure for incremental filtration updates
- O(log n) insertion time per simplex
- Fast coface retrieval for boundary matrix computation
- Index for O(1) simplex lookup by vertices

**PersistentHomology:**
- Builds boundary matrix ∂ where ∂[j,i] = 1 if simplex_i is a face of simplex_j
- Reduces matrix using Gaussian elimination (column-wise)
- Extracts birth-death intervals from low-to-high pairs
- Handles infinite intervals (classes that never die)

---

## 3. Feature Extraction and Integration

### 3.1 Feature Extraction Module

**BettiNumberExtractor:**
- Extracts Betti numbers at multiple filtration scales
- Returns vectors: β₀(ε₁), β₀(ε₂), ... for each dimension
- Computes summary statistics: max, mean, std, final value

**PersistenceFeatureExtractor:**
- Extracts scalar features from persistence diagrams
- Features per dimension: count, total_persistence, mean_persistence, max_persistence, entropy
- Global features: total_intervals, infinite_intervals

**ThreadMaturityScorer:**
- Computes maturity scores for GLN threads (thread-level, not document-level)
- Components:
  - `connectedness`: Fraction of documents connected via edges
  - `topological_stability`: Variance in Betti numbers across temporal windows
  - `persistence_plateau`: Whether total persistence has plateaued (slope near 0)
  - `dimensional_shift`: Detection of β₀ → β₁ transition
- Combines into single maturity_score (0-1, higher = more mature)

### 3.2 pn-autoresearch Integration

**TDAScorer Plugin:**
- Implements pn-autoresearch scorer interface
- Scores chunks based on pre-computed TDA features
- Supports feature caching in SignalDB for performance
- Configurable via experiment.yml

**Configuration Options:**
```yaml
scorers:
  - name: tda_features
    weight: 0.3
    # Feature selection
    betti_0_enabled: true
    betti_0_weight: 0.5
    betti_1_enabled: true
    betti_1_weight: 0.3
    betti_2_enabled: false
    max_dimension: 2

    # Persistence features
    persistence_entropy_enabled: true
    persistence_entropy_weight: 0.4
    total_persistence_enabled: true
    total_persistence_weight: 0.3

    # Thread maturity
    maturity_score_enabled: true
    maturity_score_weight: 0.5

    # Computation
    cache_features: true         # Pre-compute and cache in SignalDB
    epsilon_max: 0.7             # Distance threshold for VR complex
    num_scales: 10               # For Betti number vectors
```

### 3.3 Output Format

**Feature Storage Schema (SQL):**
```sql
CREATE TABLE tda_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    dimension INTEGER,
    feature_name TEXT NOT NULL,
    feature_value REAL NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
```

**JSON Output Format:**
```json
{
  "doc_id": "26033/282122TMAR26",
  "tda_features": {
    "betti_numbers": {
      "scales": [0.0, 0.1, 0.2, ..., 0.7],
      "betti_0": [1, 1, 3, 5, 4, 3, 2, 1, 1, 1],
      "betti_1": [0, 0, 0, 1, 2, 2, 1, 1, 1, 1]
    },
    "persistence_features": {
      "dim_0_count": 10,
      "dim_0_total_persistence": 3.45,
      "dim_0_mean_persistence": 0.345,
      "dim_0_entropy": 0.72
    },
    "maturity_score": {
      "maturity_score": 0.78,
      "connectedness": 0.85,
      "topological_stability": 0.72
    }
  }
}
```

---

## 4. Testing and Validation

### 4.1 Synthetic Topology Validation

Tests on data with known homology:

**Circle (S¹):**
- Known: β₀=1 (one component), β₁=1 (one loop)
- Test: Sample points from unit circle, compute PH
- Assert: Correct Betti numbers identified

**Torus (T²):**
- Known: β₀=1, β₁=2 (two loops), β₂=1 (one void)
- Test: Sample points from torus surface
- Assert: Correct topology detected (with tolerance for noise)

**Two Circles (S¹ ⊔ S¹):**
- Known: β₀=2 (disconnected), β₁=2 (two loops)
- Test: Two separate circles, far apart
- Assert: Disconnected components detected

### 4.2 Real Corpus Validation

**Sprint 26033 Test:**
- Load from obsidian-refs.json fixture
- Run full TDA pipeline
- Verify: No errors, reasonable output
- Test: Reproducibility (same input → same output)
- Test: Temporal consistency (features evolve smoothly)

**Sprint 26034 Test:**
- Validate on second sprint
- Compare features to 26033
- Verify: Can detect differences between sprints

### 4.3 Performance Benchmarks

**Streaming vs Batch:**
- Measure: Batch recomputation time
- Measure: Incremental update time
- Assert: Incremental is > 2x faster
- Verify: O(log n) complexity for single simplex insertion

**Memory Usage:**
- Track: Memory usage vs corpus size
- Assert: Linear growth (not exponential)
- Target: < 500MB for single sprint

### 4.4 pn-autoresearch Integration Test

**End-to-End NDCG Validation:**
- Run baseline experiment (no TDA)
- Run TDA experiment
- Compare: NDCG@10 scores
- Assert: TDA does not significantly degrade search quality (Δ > -0.02)
- Goal: Δ > 0.01 (improvement)

---

## 5. Implementation Roadmap

### 5.1 Phase Overview

**Phase 1: Python Prototype (Week 1-2)**
- Core algorithms (synthetic validation)
- Data adapters (obsidian-refs.json, SignalDB)
- Real corpus testing (26033/26034)
- Initial feature extraction

**Phase 2: pn-autoresearch Integration (Week 3)**
- TDA scorer plugin
- experiment.yml configuration
- NDCG@10 validation
- Iteration loop (optimize weights)

**Phase 3: Streaming Validation (Week 4)**
- Incremental Simplex Tree
- Performance benchmarking
- Batch vs streaming comparison
- Proof of O(log n) updates

**Phase 4: Rust Core (Week 5-8) - Optional**
- Profile Python bottlenecks
- Implement Simplex Tree in Rust
- PyO3 bindings
- Integration testing

### 5.2 Week-by-Week Breakdown

**Week 1: Core Algorithms**
- Day 1: Project scaffolding
- Day 2: Graph abstraction, ObsidianRefsGraph
- Day 3: Filtration builder, distance matrix
- Day 4: Simplex Tree data structure
- Day 5: Persistent homology, matrix reduction
- Day 5: Synthetic validation (circle, torus, two circles)

**Week 2: Real Corpus + Features**
- Day 6: SignalDB adapter
- Day 7: Real corpus testing (26033/26034)
- Day 8: Betti extraction
- Day 9: Persistence features
- Day 10: Thread maturity scoring
- Day 10: Reproducibility tests

**Week 3: pn-autoresearch Integration**
- Day 11: TDA scorer plugin
- Day 12: Feature caching (tda_features table)
- Day 13: Config integration (experiment_tda.yml)
- Day 14: Baseline experiment
- Day 15: TDA experiment
- Day 15-17: Iteration loop (optimize weights)
- Day 18: Documentation (RESEARCH.md)

**Week 4: Streaming Validation**
- Day 19: Incremental insertion, deletion
- Day 20: Local recomputation (affected region)
- Day 21: Performance benchmark
- Day 22: Memory profiling
- Day 23: Optimization
- Day 24: Documentation
- Day 25: Validation tests

### 5.3 Milestones and Decision Points

**Milestone 1: Prototype Validation** (End of Week 2)
- **Question**: Does TDA compute correct persistent homology on real data?
- **Go**: All synthetic tests pass, real corpus runs without errors
- **No-go**: Fundamental algorithm issues, rethink approach

**Milestone 2: Search Quality Impact** (End of Week 3)
- **Question**: Do TDA features improve search relevance?
- **Go**: ΔNDCG@10 > 0.01, features provide value
- **No-go**: No improvement or degradation, pivot or stop

**Milestone 3: Streaming Viability** (End of Week 4)
- **Question**: Is incremental update feasible and performant?
- **Go**: > 2x speedup, O(log n) complexity confirmed
- **No-go**: No performance gain, batch-only is fine

**Milestone 4: Rust Optimization** (End of Week 8)
- **Question**: Is Rust implementation worth the complexity?
- **Go**: > 10x speedup, memory usage acceptable
- **No-go**: Python is fast enough, skip Rust

---

## 6. Risks and Alternatives

### 6.1 Technical Risks

**Risk 1: Computational Complexity Explosion**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: Epsilon threshold, sparse approximation, graph-based PH fallback
- **Trigger**: >100k simplices, >5 minutes runtime, >1GB memory

**Risk 2: Incorrect Distance Metric**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: A/B test metrics, human validation, configurable metric
- **Trigger**: No NDCG improvement, uniform distance distribution

**Risk 3: Numeric Instability**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**: Exact arithmetic, tolerance thresholds, GUDHI validation
- **Trigger**: Inconsistent outputs, negative persistence

**Risk 4: Meaningless Maturity Scores**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Human validation, correlation analysis, document-level fallback
- **Trigger**: Uniform distribution, no human correlation

### 6.2 Project Risks

**Risk 5: No Search Improvement**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: Rapid validation in Week 2, corpus analytics fallback
- **Trigger**: Low feature-relevance correlation, zero weight optimization

**Risk 6: Integration Complexity**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**: Early test, API review, simplified prototype
- **Trigger**: Can't load scorer, type mismatches

**Risk 7: Timeline Overrun**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Conservative estimates, Rust optional, milestone gating
- **Trigger**: Week 2 tasks incomplete by Week 3

### 6.3 Alternative Approaches

**Alternative A: Graph-Based Persistent Homology**
- Compute homology directly on graph (O(|V| + |E|))
- Much faster, but less expressive (only β₀, β₁)
- Use when: VR complex becomes intractable

**Alternative B: Embedding-Based Approach**
- Use UMAP/t-SNE + HDBSCAN clustering
- Well-established, scales to millions
- Use when: TDA features don't add value

**Alternative C: Temporal Only**
- Focus on temporal evolution, skip spatial topology
- Simpler, faster
- Use when: Spatial structure is weak

**Alternative D: Use Existing Library**
- Don't implement from scratch, use GUDHI/Ripser
- Battle-tested, faster (C++)
- Use when: From-scratch has bugs or performance issues

---

## 7. Documentation and Handoff

### 7.1 Documentation Structure

```
pn-tda/
├── README.md
├── docs/
│   ├── DESIGN.md
│   ├── API.md
│   ├── INSTALL.md
│   ├── TUTORIAL.md
│   ├── RESEARCH.md
│   ├── ALTERNATIVES.md
│   ├── BENCHMARKS.md
│   └── MIGRATION.md
├── python/pn_tda/
│   └── **/*.py (all with docstrings)
├── tests/
│   ├── synthetic/README.md
│   ├── real_corpus/README.md
│   └── integration/README.md
└── examples/
    └── pn_autoresearch_integration/README.md
```

### 7.2 Session Context

**This design is based on:**
1. **Grok's Proposal** (290256TMAR26 Grok Answer.md)
   - Three improvement ideas: Incremental streaming PH, Topological e-processes, Novelty-guided filtration
   - Petri Net generalization for distributed transaction flow

2. **NotebookLM Response** (290410TMAR26 NotebookLM response.md)
   - Recommendation: Integrate TDA/Persistent Homology into CASS/cm system
   - Focus on "Agentic Memory" time-decay problem

3. **Original TDA Proposal** (282306TMAR26 TDA and Filtrations.md)
   - Filtration mathematics applied to Zettelkasten growth
   - GLN (Generalized Luhmann Numbers) as structural pointers

4. **Open Planter Schema** (pn-monorepo docs/open-planter-artifacts)
   - Graph structure: Files, GLNs, SKUs, Beads, Sprints, Terms, Notes
   - Relationships: REFERENCES, IN_SPRINT, MENTIONS, LINKS_TO, UPLINKS_TO

### 7.3 Handoff Checklist

**For the Implementing Agent:**

**Prerequisites:**
- [ ] Access to `/mnt/c/Users/jegoo/environment/pn-monorepo`
- [ ] Python 3.10+ installed
- [ ] pytest installed
- [ ] (Optional) Rust toolchain for Phase 4

**Setup:**
1. [ ] Create `pn-tda/` directory in `pn-monorepo/projects/`
2. [ ] Create `pyproject.toml` with dependencies
3. [ ] Initialize pytest configuration
4. [ ] Create test fixtures from 26033 data

**Phase 1 Deliverables:**
- [ ] All synthetic topology tests pass
- [ ] TDA pipeline runs on 26033 fixture
- [ ] Betti numbers and persistence features extracted
- [ ] Thread maturity scores computed
- [ ] Test coverage > 80%

**Phase 2 Deliverables:**
- [ ] TDA scorer integrates with pn-autoresearch
- [ ] At least one configuration shows non-negative NDCG impact
- [ ] Best configuration documented
- [ ] Results recorded in results.tsv

**Phase 3 Deliverables:**
- [ ] Incremental updates implemented
- [ ] Benchmarks show > 2x speedup
- [ ] O(log n) complexity confirmed

**Phase 4 Deliverables (Optional):**
- [ ] Rust implementation passes all tests
- [ ] PyO3 bindings expose core API
- [ ] > 10x speedup over Python
- [ ] Installation documented

---

## Appendix: Clarifying Questions Summary

This design emerged from the following clarifying questions:

**Q1: Target application?**
- **A**: Part 1 - General TDA library for pn-monorepo
- **B**: Part 2 - CASS/cm integration (future work)

**Q2: Primary use case?**
- **B**: Topological feature extraction for search (dolt-retrieve enhancement)

**Q3: Technology preference?**
- **C**: Hybrid (Rust core + Python bindings)
- **D**: Python prototype first (evaluation)

**Q4: Prototype scope?**
- **B**: Real corpus prototype (26033/26034)
- **D**: Streaming validation

**Q5: Data structure for simplicial complex?**
- **B**: Weighted distance graph → Vietoris-Rips
- **C**: Temporal filtration

**Q6: Data ingestion?**
- **D**: Flexible abstraction (obsidian-refs.json, SignalDB, Neo4j CSV)

**Q7: Output format?**
- **D**: All formats (persistence diagrams, Betti vectors, features)

**Q8: Maturity score scope?**
- Thread-level (GLN paths), not document-level

**Q9: Prototype completion criteria?**
- **C**: Search integration demo using pn-autoresearch

**Q10: TDA feature configurability?**
- **D**: Full flexibility with sensible defaults

**Q11: Data source for prototype?**
- Use Open Planter output from pn-monorepo

**Q12: Filtration construction approach?**
- **B**: Weighted distance graph → Vietoris-Rips
- **C**: Temporal filtration (research focus)

**Q13: Output for dolt-retrieve?**
- **D**: All formats (configurable)

**Q14: pn-autoresearch integration scope?**
- Full TDA feature configurability in experiment.yml

**Q15: Feature caching strategy?**
- Pre-compute and cache in SignalDB for performance

---

## References

- Boissonnat et al. (2016). "The Simplex Tree: An Efficient Data Structure for General Simplicial Complexes." arXiv:1607.08449
- Edelsbrunner & Morozov (2021). "Persistent Homology."
- Grok (2026). "Incremental Streaming Persistent Homology Engine." Twitter: @thetrollbar
- NotebookLM (2026). "TDA Integration for Agentic Memory." Google NotebookLM
- pn-monorepo (2026). azure-wiki-analysis, pn-autoresearch, Open Planter artifacts

---

**End of Design Document**

**Next Steps:**
1. Implementing agent reviews this design
2. Create implementation plan using `superpowers:writing-plans` skill
3. Begin Phase 1 (Python Prototype)
4. Report progress at each milestone

**Questions? Contact john@promptneurons.com**
