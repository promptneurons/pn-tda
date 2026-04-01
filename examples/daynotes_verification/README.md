# Daynotes Sprint Evolution

TDA analysis of the Daynotes.wiki corpus — a 15-year Azure DevOps wiki with
5,389 documents across 170 sprints (2010-2025). Three topology layers computed
per sprint, with heading content read from the corpus DB's `raw_content` column
(no vault mount needed).

## Quick Start

```bash
cd pn-tda
pip install -e ".[dev]"

# All 169 sprints, graph filtration + headings (fast, ~5s)
python3 examples/daynotes_verification/sprint_evolution.py --no-vr

# Recent 20 sprints with VR comparison
python3 examples/daynotes_verification/sprint_evolution.py --recent 20

# Cumulative (growing corpus)
python3 examples/daynotes_verification/sprint_evolution.py --no-vr --cumulative

# TSV output
python3 examples/daynotes_verification/sprint_evolution.py --no-vr --tsv > daynotes-evolution.tsv
```

## Data Sources

| Asset | Path | Description |
|-------|------|-------------|
| Signal DB | `~/azure-wiki-analysis/output/signals.db` | 5,389 docs, 1,969 signals, 4,289 edges (wiki_link + uplink + azure_devops_link) |
| Corpus DB | `~/pn-monorepo/.../kitsap-searchengine-lite/data/pn-daynotes.db` | 6,056 docs with `raw_content` for heading analysis, 12,506 chunks |

## Topology Layers

| Layer | What it captures | Source |
|-------|-----------------|--------|
| **Graph filtration** | Inter-document links (wiki_links, uplinks) | Signal DB edges, O(\|V\|+\|E\|) |
| **Vietoris-Rips** | Statistical signal co-occurrence | All-pairs Jaccard, O(n²) |
| **Heading topology** | Intra-document outline structure | Corpus DB `raw_content`, markdown `#`/`##`/`###` |

## Findings: Full 169-Sprint Longitudinal View (2010-2025)

The wiki spans two eras separated by a dormancy gap:

### Era 1: Early Daynotes (2010-2014) — 133 sprints

Flat notes with minimal linking. Most sprints have 0 edges and
maturity at the baseline floor (0.45). Heading structure is sparse.

| Sprint | Year | Docs | Edges | β₀ | β₁ | Maturity | Headings | hβ₁ |
|--------|------|------|-------|----|----|----------|----------|-----|
| 10092 | 2010-Sep | 14 | 3 | 11 | 0 | 0.54 | 2 | 0 |
| 11040 | 2011-Apr | 30 | 6 | 25 | 0 | 0.52 | 59 | 0 |
| 11052 | 2011-May | 30 | 1 | 29 | 0 | 0.46 | 62 | **22** |
| 11092 | 2011-Sep | 21 | 6 | 15 | 0 | 0.53 | 10 | 0 |
| 12090 | 2012-Sep | 70 | 7 | 63 | 0 | 0.49 | 15 | 0 |
| 13042 | 2013-Apr | 55 | 16 | 42 | 0 | 0.53 | 34 | 0 |
| 14061 | 2014-Jun | 84 | 24 | 61 | 0 | 0.53 | 277 | **48** |
| 14091 | 2014-Sep | 38 | 1 | 37 | 0 | 0.46 | 226 | **20** |
| 14102 | 2014-Oct | 50 | 17 | 33 | 0 | 0.52 | 0 | 0 |

No β₁ loops ever appear in the inter-document graph — pure tree/forest
structure. Heading loops (hβ₁) first appear at 11052 (May 2011) and peak
at 14061 (Jun 2014, 48 heading loops from 277 headings).

### Gap: 2015-2018

Single sprint `9100` (2019-Oct, 3 docs) — wiki largely dormant.

### Era 2: Active Development (2025) — 35 sprints

Knowledge graph emerges. Edges increase, β₁ appears, maturity rises sharply.

| Sprint | Year | Docs | Edges | β₀ | β₁ | Maturity | Headings | hβ₁ |
|--------|------|------|-------|----|----|----------|----------|-----|
| 25011 | 2025-Jan | 80 | 12 | 68 | 0 | 0.51 | 3 | 0 |
| 25041 | 2025-Apr | 57 | 22 | 35 | 0 | 0.52 | 17 | 0 |
| 25052 | 2025-May | 80 | 4 | 76 | 0 | 0.47 | 99 | **73** |
| 25061 | 2025-Jun | 57 | 1 | 56 | 0 | 0.46 | 150 | **45** |
| **25072** | **2025-Jul** | **129** | **41** | **94** | **4** | **0.78** | **317** | **69** |
| 25082 | 2025-Aug | 99 | 40 | 63 | **3** | **0.77** | 45 | 0 |
| 25090 | 2025-Sep | 112 | 66 | 57 | **1** | **0.76** | 164 | 4 |
| 25091 | 2025-Sep | 40 | 14 | 26 | 0 | 0.52 | 0 | 0 |

**Sprint 25072 (Jul 2025) is the topological peak**: 129 docs, 41 edges,
4 inter-document loops (β₁), maturity 0.78, and the richest heading structure
(317 headings, 69 heading loops). This is when the Zettelkasten structure
fully crystallizes.

### Trend Summary (169 sprints)

```
Graph filtration: maturity 0.450 → 0.520  conn 0.000 → 0.359
Peak maturity:    25072 (2025-Jul-2) = 0.782
Most headings:    25072 (317)
Most heading loops: 25052 (hβ₁=73)
```

### Graph Filtration vs VR (per-sprint comparison, recent 20)

VR sees every sprint as a single connected component (β₀=1) because
all-pairs Jaccard puts all documents within ε=1.0 — it detects no
topological structure. Graph filtration reveals the actual link structure:

| | Graph Filtration | Vietoris-Rips |
|---|---|---|
| **β₀ range** | 8–94 components | Always 1 |
| **β₁ range** | 0–4 loops | Always 0 |
| **Maturity trend** | 0.45 → 0.78 | 0.44 → 0.47 (flat) |
| **Simplices** | 5–190 (sparse) | 25–358K (dense) |

### Heading Topology Highlights

| Sprint | Headings | Depth | hβ₁ | Branching | Observation |
|--------|----------|-------|-----|-----------|-------------|
| 25052 | 99 | 4 | **73** | 11.6 | Highest branching — flat, wide outlines |
| 25061 | 150 | 4 | **45** | 2.8 | Rich internal structure |
| 25072 | 317 | 4 | **69** | 3.7 | Most headings — largest sprint |
| 14061 | 277 | 4 | **48** | 3.6 | Era 1 peak — STI curriculum outlines |
| 14091 | 226 | 4 | **20** | 2.2 | Student syllabi with deep structure |

### Key Differences from Obsidian-Local

| | Daynotes | Obsidian-Local |
|---|---|---|
| **Corpus age** | 15 years (2010-2025) | 1 year (Jun 2025 - Mar 2026) |
| **Documents** | 5,389 | 1,402 |
| **Sprints** | 170 | 24 |
| **Edge types** | wiki_link (624), uplink (207), azure_devops (3,458) | obsidian_wikilink (1,681), uplink (535) |
| **Edge resolution** | Slug-based (wiki paths) | Slug-based (wikilink targets) |
| **Heading source** | Corpus DB `raw_content` | Vault `.md` files |
| **Typical sprint β₁** | 0-4 | 0-14 |

Daynotes has fewer intra-wiki links per sprint (wiki_link + uplink only,
most edges are external azure_devops_link). Obsidian-local is denser because
`[[wikilinks]]` create more internal connections per document.

## Architecture

```
signals.db (Daynotes)               pn-daynotes.db (corpus)
    ↓                                       ↓
[SignalDBGraph]                    [raw_content → heading_graph]
    ↓                                       ↓
[GraphFiltrationBuilder]           [SimplexTree from heading edges]
[VietorisRipsBuilder]              [PersistentHomology → hβ₀, hβ₁]
    ↓
[PersistentHomology]
    ↓
[Betti + Persistence + Maturity]
    ↓
Per-sprint tabular output (Graph, VR, Headings side-by-side)
```

## Beads

- **bd-g9p** — TDA/Filtrations Pipeline (parent epic)
- **bd-3b6** — Corpus validation
