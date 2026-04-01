# pn-tda — Claude Code Guidance

## Purpose
Topological Data Analysis library for extracting topological features from knowledge graphs.
Part of the pn-monorepo ecosystem. See `/data/projects/tda-brainstorming-2026-03-31.md` for full design.

## Beads
- **bd-g9p** — TDA/Filtrations Pipeline (epic)
- **bd-3m2** — Core algorithms (graph, filtration, simplex tree, persistence)
- **bd-csc** — Data adapters (ObsidianRefs, SignalDB, Neo4j CSV)
- **bd-2z4** — Feature extraction (Betti, persistence, maturity)
- **bd-3b6** — Synthetic + real corpus validation
- **bd-8l5** — pn-autoresearch scorer integration

## Key Files
- `src/pn_tda/core/graph.py` — Graph ABC (all adapters implement this)
- `src/pn_tda/core/simplex_tree.py` — Simplex Tree data structure
- `src/pn_tda/core/filtration.py` — Vietoris-Rips filtration builder
- `src/pn_tda/core/persistence.py` — Persistent homology via matrix reduction

## Setup
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Conventions
- src/ layout with setuptools
- Tests in tests/ using pytest
- Synthetic topology tests validate against known Betti numbers
