# Obsidian-Local TDA Verification

Topological analysis of the obsidian-local Zettelkasten vault, leveraging
hub pages, uplink hierarchy, and classification signals as strong
topological indicators.

## Domain Signals

### Hub Pages (Focus/Foci)

Hub pages are human-curated directories — pages with many internal wikilinks
acting as topological nexus points. In obsidian-local:
- **76 edges**: CEC classification index (00_index.md)
- **56 edges**: MHD and EMP science station
- **43 edges**: JEFFLIB node list
- **39 edges**: Company operations manual

These pages encode human judgment about document proximity. A link from a hub
page is a strong signal that the target is "about" that hub's topic.

### Uplink Hierarchy

The `Uplink:` pattern creates parent-child relationships between documents:
```
Uplink: [[201013UNOV25 Sprint Planning 25112]]
```
This builds a DAG where sprint planning pages are parents of topic threads.
The obsidian-local corpus has **535 uplinks** creating a rich hierarchy.

### GLN (Generalized Luhmann Numbers)

When present, GLN labels encode human-assessed topological closeness:
- Documents sharing a GLN prefix (e.g., `1a1b`) are considered neighbors
- The GLN hierarchy mirrors the Zettelkasten branching structure
- Open Planter's EXTRACTION-SUMMARY.md and GLN-Sprint-26033-map.md provide
  curated mappings (generate by running Open Planter on the vault)

### Classification References

JDN (Jefflib Decimal Notation) numbers cluster documents by subject:
- `JDN 13` (13 docs): Education / Pedagogy
- `JDN 17` (11 docs): Arts / Literature
- `JDN 14` (9 docs): History
- `SUMO-MILO` (20 docs): Upper ontology references

## Quick Start

```bash
cd pn-tda
pip install -e ".[dev]"

# Run with auto-detected paths
python3 examples/obsidian_local_verification/demo.py

# Explicit paths
python3 examples/obsidian_local_verification/demo.py \
    --signal-db ~/azure-wiki-analysis/output/obsidian-signals.db \
    --corpus-db ~/pn-monorepo/.../kitsap-searchengine-lite/data/pn-daynotes.db \
    --wordnet-dir ~/pn-monorepo/.../data/Ontologies/wordnet-mappings
```

## Generating a Fresh Signal DB

If you have access to the obsidian-local vault:

```bash
cd azure-wiki-analysis
wiki-analyze extract \
    --config examples/obsidian-vault.yaml \
    --wiki-path ~/environment/obsidian-local \
    --db output/obsidian-local-signals.db
```

The config enables: `uplink`, `dtg`, `links`, `classification_ref`, `obsidian_link`
detectors with focus_hub pattern `(?:focus|foci|hub|registry)`.

## What the Demo Shows

1. **Hub page analysis**: Top 20 pages by internal link count, uplink hierarchy
2. **Classification clusters**: JDN/SUMO signal distribution
3. **Full corpus topology**: Betti numbers, persistence, maturity on flat signal distance
4. **Ontology-enriched topology**: Same with CEC hierarchy + SUMO/WordNet blended distance
5. **Comparison**: Side-by-side metrics showing ontology effect

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Documents | 1,402 |
| Signals | 4,739 |
| Wikilinks | 1,681 |
| Uplinks | 535 |
| Hub pages (flagged) | 3 |
| Hub pages (>10 edges) | ~20 |
| JDN classifications | 15 distinct codes |
| Top-level folders | Fleeting (1,146), Playbooks (140), Topics (28), ZK (27) |

## Architecture

```
obsidian-signals.db
    ↓
[SignalDBGraph] ─── slug resolution for wikilinks ───→ 2,601 edges
    ↓                                                        ↓
[OntologyDistance] ←── pn-daynotes.db (CEC hierarchy)   [hub analysis]
    ↓               ←── WordNetMappings (SUMO expansion)     ↓
[GraphFiltrationBuilder] ←── uplink edges as hierarchy  [uplink DAG]
    ↓
[PersistentHomology] → [Features] → Maturity score
```

## Sprint Evolution Experiment

`sprint_evolution.py` processes each sprint sequentially, computing TDA metrics
per sprint and presenting a tabular result showing knowledge graph evolution.

Three topology layers are computed side-by-side for each sprint:

| Layer | What it captures | Complexity | Source |
|-------|-----------------|------------|--------|
| **Graph filtration** | Inter-document links (wikilinks, uplinks) | O(\|V\|+\|E\|) | Signal DB edges |
| **Vietoris-Rips** | Statistical signal co-occurrence | O(n²) | All-pairs Jaccard |
| **Heading topology** | Intra-document outline structure | O(headings) | Markdown `#`/`##`/`###` |

```bash
python3 examples/obsidian_local_verification/sprint_evolution.py                                  # per-sprint
python3 examples/obsidian_local_verification/sprint_evolution.py --cumulative                      # growing corpus
python3 examples/obsidian_local_verification/sprint_evolution.py --vault-path ~/environment/obsidian-local  # with headings
python3 examples/obsidian_local_verification/sprint_evolution.py --tsv                             # machine-readable
```

### Per-Sprint Findings (Graph Filtration)

| Phase | Sprints | Pattern |
|-------|---------|---------|
| **Early** (Jun-Aug 2025) | 25061-25081 | 0 edges, β₁=0, maturity ~0.45. Pure note-taking. |
| **Linking begins** (Sep-Oct) | 25091-25102 | First edges appear, connectedness rises to 0.22 |
| **Dimensional shift** (Nov 2025) | 25112 | **First β₁ > 0** — loops appear. Maturity jumps to 0.76. Zettelkasten structure emerges. |
| **Rapid densification** (Dec-Jan) | 25120-26012 | β₁ hits 6, edges triple. Sprint 26011 has 83 edges in 74 docs. |
| **Peak connectivity** (Mar 2026) | 26031 | 82% connected, 5 loops, 2 H1 persistence intervals. Most topologically mature sprint. |

### Cumulative View (growing corpus)

| Metric | Start (Jun 2025) | End (Mar 2026) | Trend |
|--------|-------------------|-----------------|-------|
| Maturity | 0.503 | 0.756 | +50% |
| Connectedness | 0.133 | 0.534 | +4x |
| β₁ (loops) | 0 | 142 | Steady growth |
| H0 entropy | 0.69 | 6.39 | +9x (topological diversity) |
| Dimensional shift | Nov 2025 | — | First β₁>0 at sprint 25112 |

### Graph Filtration vs Vietoris-Rips

Running both builders side-by-side on per-sprint data (5-88 docs, feasible for O(n²)):

| | Graph Filtration | Vietoris-Rips |
|---|---|---|
| **What it measures** | Actual knowledge structure (wiki links + uplinks) | Statistical co-occurrence (signal Jaccard) |
| **β₀ range** | 5–61 components (real structure) | Always 1 (everything connected) |
| **β₁ loops** | 0→14 (structure emerges at 25112) | Always 0 (dense blob, no holes) |
| **Maturity trend** | 0.50 → 0.77 (+54%) | 0.47 → 0.43 (flat/declining) |
| **Simplices** | 5–196 (sparse, fast) | 25–113K (dense, slow) |

VR sees every sprint as a single fully-connected component (β₀=1, conn=1.0)
because all-pairs Jaccard puts all documents within ε=1.0 — it detects no
topological structure. Graph filtration preserves the actual link structure
that humans created. **For knowledge graphs, the graph IS the topology.**

### Heading Topology (Intra-Document Structure)

When `--vault-path` is provided, each document's markdown heading hierarchy
(`#` → `##` → `###`) is parsed into a tree and TDA is computed on the
combined heading graph per sprint.

Heading topology columns:
- **Hdg**: total headings across the sprint's documents
- **Dp**: max heading depth (1-6)
- **hβ₀**: heading graph components (separate outline trees)
- **hβ₁**: heading graph loops (sibling cross-references within outlines)
- **Br**: average branching factor (children per non-leaf heading)

Key findings:

| Sprint | Headings | Depth | hβ₁ (loops) | Branching | Observation |
|--------|----------|-------|-------------|-----------|-------------|
| 25072 | 94 | 3 | 73 | 20.2 | Highest branching — flat wide outlines |
| 25111 | 396 | 6 | 91 | 4.3 | Deepest outlines (6 levels), rich structure |
| 25120 | 474 | 4 | 79 | 18.9 | High branching + many loops |
| 26020 | 2,167 | 6 | 741 | 5.9 | **Most heading loops** — deeply structured documents with extensive cross-referencing |
| 26031 | 319 | 4 | 93 | 3.7 | Sprint 26033 plan period — dense internal structure |

Sprint 26020 (Feb 2026) stands out with 2,167 headings and 741 heading loops
— this sprint contains documents with deeply nested, richly cross-referenced
outlines. The heading topology captures document maturity that the inter-document
graph cannot see.

## Beads

- **bd-g9p** — TDA/Filtrations Pipeline (parent epic)
- **bd-csc** — Data adapters (SignalDBGraph with slug resolution)
- **bd-3b6** — Corpus validation
