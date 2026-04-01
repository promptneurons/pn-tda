# Ontology-Enriched TDA

Demonstrates how ontology-aware distance changes the topological features of a
document knowledge graph, comparing flat signal Jaccard against a blended distance
that incorporates CEC/JDN concept hierarchy and SUMO/WordNet semantic expansion.

## What This Shows

The demo runs the full TDA pipeline twice on the same graph:

1. **Flat signal distance** — Jaccard on `{signal_type:signal_value}` sets (baseline)
2. **Ontology-enriched distance** — weighted blend of:
   - **Hierarchy distance (40%)**: CEC/JDN concept tree ancestry overlap (documents
     sharing a parent like `CEC B5 → B51 → B512` are closer)
   - **SUMO distance (30%)**: WordNet → SUMO upper-ontology expansion. Words in
     document content are mapped through 147k WordNet lemmas to 112k SUMO concepts
     (e.g., "war" → `WarfareSituation`, "tax" → `Tax`)
   - **Signal distance (30%)**: existing classification ref co-occurrence

Then compares: Betti numbers, persistence features, maturity scores, and sample
edge distances.

## Data Sources

| Source | What it provides | Size |
|--------|-----------------|------|
| `signals.db` | Document graph (5,389 nodes, 2,601 edges) | 4.1 MB |
| `pn-daynotes.db` | CEC concept hierarchy (133 concepts, 75 links, 10k concept tags) | 96 MB |
| `wordnet-mappings/` | WordNet 3.0 → SUMO bridge (noun + verb + adj) | 22 MB |

## Quick Start

```bash
cd pn-tda
pip install -e ".[dev]"

# Run with auto-detected paths
python3 examples/ontology_enriched_tda/demo.py

# Explicit paths
python3 examples/ontology_enriched_tda/demo.py \
    --signal-db ~/azure-wiki-analysis/output/signals.db \
    --corpus-db ~/pn-monorepo/.../kitsap-searchengine-lite/data/pn-daynotes.db \
    --wordnet-dir ~/pn-monorepo/.../data/Ontologies/wordnet-mappings
```

## Example Output

```
══════════════════════════════════════════════════
  COMPARISON: Flat vs Ontology-Enriched
══════════════════════════════════════════════════
  Metric                              Flat   Ontology        Δ
  ───────────────────────────────────────────────────────────────
  Maturity score                    0.7707     0.7742   +0.0035
  Connectedness                     0.9995     0.9995   +0.0000
  Dim 0 persistence entropy         4.3284     4.3301   +0.0017
  β₀ final                              3          3       +0
  β₁ final                              5          7       +2

  Distance Comparison (sample edges)
  ──────────────────────────────────────────────────────────────────────
  Pair                                 Signal   Ontology        Δ
  ab3f92c1..↔7e1d4a08..             0.9412     0.7647   -0.1765
```

## Architecture

```
signals.db (edges + signals)       pn-daynotes.db (concept hierarchy)
         ↓                                    ↓
   [SignalDBGraph]                  [OntologyDistance]
         ↓                                    ↓
    graph.edges() ──────→ [GraphFiltrationBuilder]
                             ↓
              get_distance(u, v) = 0.4 * hierarchy
                                 + 0.3 * SUMO
                                 + 0.3 * signal
                             ↓
                    [SimplexTree] → [PersistentHomology]
                             ↓
              [Betti + Persistence + Maturity features]
```

## Ontology Sources

- **CEC hierarchy** (`cec-hierarchy.ttl`): Cutter Expansive Classification (1882),
  26 main classes A-Z with 5-level subdivisions. Loaded into corpus DB by
  `kitsap-searchengine-lite/load_ontology.py`.

- **SUMO/WordNet** ([ontologyportal/sumo](https://github.com/ontologyportal/sumo/tree/master/WordNetMappings)):
  Suggested Upper Merged Ontology mapped to WordNet 3.0 synsets. Relationship types:
  `=` equivalent, `+` subsumed by, `@` instance of.

- **MAS ontology** (`mas-ontology.ttl`): Markos Analytics Suite domain ontology
  covering Financial, Health, Nutrition, Planning domains.

## Beads

- **bd-g9p** — TDA/Filtrations Pipeline (parent epic)
- **bd-3m2** — Core algorithms
- **bd-csc** — Data adapters
