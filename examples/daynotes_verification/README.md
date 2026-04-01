# Daynotes Sprint Evolution

TDA analysis of the Daynotes.wiki corpus — a 15-year Azure DevOps wiki with
5,389 documents across 170 sprints (2010-2025). Three topology layers computed
per sprint, with heading content read from the corpus DB's `raw_content` column
(no vault mount needed).

## Quick Start

```bash
cd pn-tda
pip install -e ".[dev]"

# Recent 20 sprints (fast)
python3 examples/daynotes_verification/sprint_evolution.py --recent 20

# All 170 sprints
python3 examples/daynotes_verification/sprint_evolution.py

# Cumulative (growing corpus)
python3 examples/daynotes_verification/sprint_evolution.py --cumulative --recent 20

# TSV output
python3 examples/daynotes_verification/sprint_evolution.py --tsv > daynotes-evolution.tsv
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

## Findings (Recent 20 Sprints, Mar-Sep 2025)

### Graph Filtration vs VR

Same story as obsidian-local: VR sees every sprint as a single connected
component (β₀=1). Graph filtration reveals actual structure:

| Sprint | Docs | Edges | GF β₀ | GF β₁ | GF Maturity | VR β₀ |
|--------|------|-------|-------|-------|-------------|-------|
| 25030 | 20 | 7 | 15 | 0 | 0.53 | 1 |
| 25041 | 57 | 22 | 35 | 0 | 0.52 | 1 |
| 25072 | 129 | 41 | 94 | **4** | **0.78** | 1 |
| 25082 | 99 | 40 | 63 | **3** | **0.77** | 1 |
| 25090 | 112 | 66 | 57 | **1** | **0.76** | 1 |

**Peak maturity: Sprint 25072 (Jul 2025)** — 129 docs, 41 edges, 4 loops, 0.78 maturity.
This is the most topologically mature sprint in the recent window.

### Heading Topology Highlights

| Sprint | Headings | Depth | hβ₁ (loops) | Branching | Observation |
|--------|----------|-------|-------------|-----------|-------------|
| 25052 | 99 | 4 | **73** | 11.6 | Highest branching — flat, wide outlines with many sibling cross-refs |
| 25061 | 150 | 4 | **45** | 2.8 | Rich internal structure |
| 25072 | 317 | 4 | 69 | 3.7 | **Most headings** — largest sprint also has deepest outlines |
| 25071 | 129 | 4 | 10 | 4.5 | Well-structured but fewer cross-references |

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
