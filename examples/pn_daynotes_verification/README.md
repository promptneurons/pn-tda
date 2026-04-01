# PN Daynotes NDCG Verification

End-to-end verification of TDA features against the real PN Daynotes corpus,
using the pn-autoresearch evaluation framework.

## Overview

This experiment measures whether adding TDA topological features to the
kitsap-searchengine-lite scoring pipeline improves search quality (NDCG@10)
on the PN Daynotes wiki corpus.

**Pipeline:**
```
signals.db (azure-wiki-analysis)
    → SignalDBGraph adapter
    → VietorisRips filtration
    → Persistent Homology
    → Feature extraction (Betti, persistence, maturity)
    → TDAScorer (per-document composite score)
    → Fusion with keyword Jaccard baseline
    → NDCG@10 evaluation against pinned queries
```

## Prerequisites

| Asset | Expected Path | Description |
|-------|--------------|-------------|
| Corpus DB | `$CORPUS_DB` or `~/pn-monorepo/pn-monorepo/projects/kitsap-searchengine-lite/data/pn-daynotes.db` | SQLite with chunks + documents tables (12,506 chunks) |
| Signal DB | `$SIGNAL_DB` or `~/azure-wiki-analysis/output/signals.db` | azure-wiki-analysis output for Daynotes wiki |
| Queries | `$QUERIES_PATH` or `pn-autoresearch/queries/pn-daynotes.jsonl` | 25 queries with human relevance judgments |
| pn-tda | installed (`pip install -e ".[dev]"`) | This library |

**Optional (for Obsidian vault instead of Daynotes):**
- `~/azure-wiki-analysis/output/obsidian-signals.db` (1,402 docs, 4,739 signals)

## Quick Start

```bash
cd pn-tda

# Install pn-tda if not already
pip install -e ".[dev]"

# Run with defaults (auto-detects paths)
python3 examples/pn_daynotes_verification/run_verification.py

# Or specify paths explicitly
python3 examples/pn_daynotes_verification/run_verification.py \
    --corpus-db ~/pn-monorepo/pn-monorepo/projects/kitsap-searchengine-lite/data/pn-daynotes.db \
    --signal-db ~/azure-wiki-analysis/output/signals.db \
    --queries ~/pn-monorepo/projects/pn-autoresearch/queries/pn-daynotes.jsonl

# Use obsidian signals instead
python3 examples/pn_daynotes_verification/run_verification.py \
    --signal-db ~/azure-wiki-analysis/output/obsidian-signals.db
```

## What It Does

1. **Loads** the corpus DB and signal DB
2. **Runs baseline** search: keyword Jaccard only → NDCG@10
3. **Runs TDA pipeline** on signal DB: VR complex → PH → features
4. **Runs TDA experiment**: keyword Jaccard + TDA composite (configurable weight) → NDCG@10
5. **Reports** ΔNDCG@10 with KEEP/DISCARD/NEUTRAL decision
6. **Outputs** TSV row compatible with pn-autoresearch results.tsv

## Configuration

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--corpus-db` | `CORPUS_DB` | auto-detect | Path to corpus SQLite |
| `--signal-db` | `SIGNAL_DB` | auto-detect | Path to signal SQLite |
| `--queries` | `QUERIES_PATH` | auto-detect | Path to queries JSONL |
| `--tda-weight` | `TDA_WEIGHT` | 0.3 | TDA scorer weight in fusion (0-1) |
| `--epsilon-max` | `TDA_EPSILON_MAX` | 1.0 | VR complex distance threshold |
| `--max-dimension` | `TDA_MAX_DIM` | 2 | Maximum simplex dimension |
| `--num-scales` | `TDA_NUM_SCALES` | 10 | Filtration scale resolution |

## Expected Output

```
============================================================
PN Daynotes NDCG Verification
============================================================

Corpus:  pn-daynotes.db (6056 docs, 12506 chunks)
Signals: signals.db (5389 docs, 1969 signals)
Queries: pn-daynotes.jsonl (25 queries)

1. Running BASELINE (keyword Jaccard only)...
2. Running TDA pipeline...
   Graph: 5389 nodes
   VR complex: ... simplices
   Persistence intervals: ...
   Features: maturity=..., entropy=..., stability=...
3. Running TDA experiment (keyword 70% + TDA 30%)...

RESULTS
----------------------------------------------------
Metric               Baseline        TDA      Delta
NDCG@10                0.XXXX     0.XXXX    +0.XXXX
MRR                    0.XXXX     0.XXXX    +0.XXXX
P@3                    0.XXXX     0.XXXX    +0.XXXX

ΔNDCG@10 = +X.XXXX  →  KEEP/DISCARD/NEUTRAL
```

## Interpreting Results

- **KEEP** (ΔNDCG > +0.01): TDA features improve search quality
- **NEUTRAL** (-0.02 ≤ ΔNDCG ≤ +0.01): No significant change
- **DISCARD** (ΔNDCG < -0.02): TDA features hurt search quality

If NEUTRAL or DISCARD, try:
- Different `--tda-weight` (lower = less TDA influence)
- Different `--epsilon-max` (controls graph connectivity)
- Obsidian signal DB instead of Daynotes (richer signals)

## Beads

- **bd-1cu** — NDCG validation experiment
- **bd-g9p** — TDA/Filtrations Pipeline (parent epic)
