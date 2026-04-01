# pn-tda

Topological Data Analysis library for knowledge graphs in pn-monorepo.

Computes persistent homology on document corpora using Vietoris-Rips complexes,
extracts topological features (Betti numbers, persistence diagrams, thread maturity scores),
and integrates with pn-autoresearch as a search quality scorer.

## Setup

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest tests/ -v
```

## Project Structure

- `src/pn_tda/core/` — Graph abstraction, filtration, simplex tree, persistent homology
- `src/pn_tda/adapters/` — Data adapters (ObsidianRefs, SignalDB)
- `src/pn_tda/features/` — Feature extraction (Betti numbers, persistence, maturity)
- `src/pn_tda/utils/` — Distance metrics and helpers
- `tests/synthetic/` — Validation against known topologies (circle, torus, etc.)
