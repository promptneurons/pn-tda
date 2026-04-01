# TDA Scorer — pn-autoresearch Integration Example

This example demonstrates how to use pn-tda topological features as a scorer
in the kitsap-searchengine-lite search engine, integrated via the
pn-autoresearch experiment loop.

## Overview

The `TDAScorer` computes topological features from a SignalDB (azure-wiki-analysis
output) and uses them to score document chunks. It follows the `Scorer` interface
from kitsap-searchengine-lite, so it can be dropped into the existing scorer
registry and fused with other scorers (keyword, FTS5, concept tags, etc.).

**Scoring components:**

| Component | What it measures | Weight (default) |
|-----------|-----------------|-----------------|
| `maturity_score` | Thread maturity: connectedness, stability, plateau, dimensional shift | 0.5 |
| `persistence_entropy` | Diversity of topological features (higher = richer structure) | 0.3 |
| `betti_stability` | Low variance in connected components across scales | 0.2 |

## Quick Start

### 1. Run the demo

```bash
cd pn-tda
pip install -e ".[dev]"
python examples/pn_autoresearch_integration/demo.py
```

This creates small test databases in-memory, runs the full TDA pipeline, and
prints scored chunks with feature values.

### 2. Run the test

```bash
pytest tests/test_tda_scorer.py -v
```

## Integration with kitsap-searchengine-lite

### Step 1: Copy the scorer

```bash
cp examples/pn_autoresearch_integration/tda_scorer.py \
   /path/to/kitsap-searchengine-lite/src/scorers/tda.py
```

Then edit the import at the top to use the real interface:

```python
# Replace the stub imports:
#   from examples... import Scorer, ScoredChunk
# With:
from src.scorer import Scorer, ScoredChunk
```

### Step 2: Register the scorer

Add to `kitsap-searchengine-lite/src/scorers/__init__.py`:

```python
from src.scorers.tda import TDAScorer  # noqa: F401
```

### Step 3: Add to experiment config

Copy `experiment_tda.yml` to your config directory and adjust:

```yaml
scorers:
  - name: tda_features
    weight: 0.30
    signal_db: "/path/to/your/signals.db"
    epsilon_max: 0.7
    # ... other params
```

### Step 4: Run with pn-autoresearch

```bash
cd pn-autoresearch
python experiment.py --config config/experiment_tda.yml --db /path/to/corpus.db
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal_db` | str | `$TDA_SIGNAL_DB` | Path to azure-wiki-analysis SQLite database |
| `epsilon_max` | float | 0.7 | Maximum distance for VR complex construction |
| `max_dimension` | int | 2 | Maximum simplex dimension (0=nodes, 1=edges, 2=triangles) |
| `num_scales` | int | 10 | Number of filtration scales for Betti computation |
| `maturity_weight` | float | 0.5 | Weight for maturity score in composite |
| `persistence_weight` | float | 0.3 | Weight for persistence entropy in composite |
| `betti_weight` | float | 0.2 | Weight for Betti stability in composite |

The `signal_db` path can also be set via the `TDA_SIGNAL_DB` environment variable.

## Architecture

```
azure-wiki-analysis output (signals.db)
        ↓
  [SignalDBGraph adapter]
        ↓
  [VietorisRipsBuilder] → Simplicial complex
        ↓
  [PersistentHomology] → Persistence intervals
        ↓
  [BettiExtractor + PersistenceExtractor + MaturityScorer]
        ↓
  Per-document composite score (cached)
        ↓
  [TDAScorer.retrieve()] → ScoredChunk list
        ↓
  [RRF/weighted_sum fusion] with other scorers
        ↓
  Final ranked results → NDCG@10 evaluation
```

## Beads

- **bd-8l5** — pn-autoresearch scorer integration (this example)
- **bd-g9p** — TDA/Filtrations Pipeline (parent epic)
- **bd-1cu** — NDCG validation experiment (next step)
