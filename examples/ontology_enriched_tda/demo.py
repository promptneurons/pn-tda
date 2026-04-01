#!/usr/bin/env python3
"""Demo: Ontology-Enriched TDA on the PN Daynotes Corpus.

Shows how ontology-aware distance (CEC hierarchy + SUMO/WordNet expansion)
changes the topological features of a document graph compared to flat
signal Jaccard alone.

Usage:
    cd pn-tda
    python3 examples/ontology_enriched_tda/demo.py
    python3 examples/ontology_enriched_tda/demo.py --wordnet-dir /path/to/WordNetMappings
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import GraphFiltrationBuilder
from pn_tda.core.persistence import PersistentHomology, betti_numbers
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor
from pn_tda.utils.ontology import OntologyDistance

WORD_RE = re.compile(r"\b[a-z]{3,}\b")

DEFAULT_SIGNAL_PATHS = [
    os.path.expanduser("~/azure-wiki-analysis/output/signals.db"),
    os.path.expanduser("~/azure-wiki-analysis/output/obsidian-signals.db"),
]
DEFAULT_CORPUS_PATHS = [
    os.path.expanduser(
        "~/pn-monorepo/pn-monorepo/projects/kitsap-searchengine-lite/data/pn-daynotes.db"
    ),
]
DEFAULT_WORDNET_PATHS = [
    os.path.expanduser("~/pn-monorepo/pn-monorepo/data/Ontologies/wordnet-mappings"),
    str(Path(__file__).resolve().parents[4] / "data/Ontologies/wordnet-mappings"),
]


def find_path(candidates: list[str], label: str) -> str | None:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def run_pipeline(graph, label: str, max_dimension: int = 2, num_scales: int = 10):
    """Run graph filtration + PH + feature extraction, print results."""
    doc_ids = list(graph.nodes())
    edges = list(graph.edges())

    t0 = time.time()
    builder = GraphFiltrationBuilder(max_dimension=max_dimension)
    st = builder.build(graph)
    intervals = PersistentHomology().compute(st)
    pipeline_time = time.time() - t0

    # Extract features
    betti = BettiNumberExtractor().extract(
        intervals, num_scales=num_scales, epsilon_max=1.0, max_dimension=max_dimension,
    )
    persist = PersistenceFeatureExtractor().extract(intervals, max_dimension=max_dimension)
    maturity = ThreadMaturityScorer().score(
        intervals, graph, num_scales=num_scales, epsilon_max=1.0,
    )

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Nodes: {len(doc_ids)}, Edges: {len(edges)}")
    print(f"  Simplices: {st.num_simplices()}, Intervals: {len(intervals)}")
    print(f"  Pipeline: {pipeline_time:.2f}s")
    print()
    print(f"  Betti numbers (final scale):")
    print(f"    β₀ = {betti['summary']['betti_0_final']}  (connected components)")
    print(f"    β₁ = {betti['summary']['betti_1_final']}  (loops)")
    if max_dimension >= 2:
        print(f"    β₂ = {betti['summary']['betti_2_final']}  (voids)")
    print()
    print(f"  Betti stability: β₀ max={betti['summary']['betti_0_max']}, "
          f"mean={betti['summary']['betti_0_mean']:.1f}, "
          f"std={betti['summary']['betti_0_std']:.1f}")
    print()
    print(f"  Persistence (dim 0):")
    print(f"    Count: {persist['dim_0_count']}")
    print(f"    Total: {persist['dim_0_total_persistence']:.4f}")
    print(f"    Mean:  {persist['dim_0_mean_persistence']:.4f}")
    print(f"    Max:   {persist['dim_0_max_persistence']:.4f}")
    print(f"    Entropy: {persist['dim_0_entropy']:.4f}")
    if persist['dim_1_count'] > 0:
        print(f"  Persistence (dim 1):")
        print(f"    Count: {persist['dim_1_count']}")
        print(f"    Total: {persist['dim_1_total_persistence']:.4f}")
        print(f"    Entropy: {persist['dim_1_entropy']:.4f}")
    print()
    print(f"  Maturity score:  {maturity['maturity_score']:.4f}")
    print(f"    Connectedness:         {maturity['connectedness']:.4f}")
    print(f"    Topological stability: {maturity['topological_stability']:.4f}")
    print(f"    Persistence plateau:   {maturity['persistence_plateau']:.4f}")
    print(f"    Dimensional shift:     {maturity['dimensional_shift']:.4f}")

    return {
        "betti": betti, "persist": persist, "maturity": maturity,
        "simplices": st.num_simplices(), "intervals": len(intervals),
        "time": pipeline_time,
    }


def sample_distance_pairs(graph, ont_graph, n_pairs: int = 10):
    """Show how ontology distance differs from flat signal distance for sample pairs."""
    edges = list(graph.edges())
    if not edges:
        return

    print(f"\n{'─' * 50}")
    print(f"  Distance Comparison (sample edges)")
    print(f"{'─' * 50}")
    print(f"  {'Pair':<40} {'Signal':>8} {'Ontology':>10} {'Δ':>8}")
    print(f"  {'─' * 68}")

    shown = 0
    for src, tgt in edges:
        if shown >= n_pairs:
            break
        d_signal = graph.get_distance(src, tgt)
        d_ontology = ont_graph.get_distance(src, tgt)
        delta = d_ontology - d_signal

        # Show doc_id abbreviated
        src_short = src[:8]
        tgt_short = tgt[:8]
        print(f"  {src_short}..↔{tgt_short}.."
              f"  {d_signal:>8.4f} {d_ontology:>10.4f} {delta:>+8.4f}")
        shown += 1


def main():
    parser = argparse.ArgumentParser(description="Ontology-Enriched TDA Demo")
    parser.add_argument("--signal-db", help="Path to signal SQLite")
    parser.add_argument("--corpus-db", help="Path to corpus SQLite (for concept hierarchy)")
    parser.add_argument("--wordnet-dir", help="Path to WordNetMappings directory")
    args = parser.parse_args()

    signal_path = args.signal_db or find_path(DEFAULT_SIGNAL_PATHS, "signal DB")
    corpus_path = args.corpus_db or find_path(DEFAULT_CORPUS_PATHS, "corpus DB")
    wordnet_dir = args.wordnet_dir or find_path(DEFAULT_WORDNET_PATHS, "WordNet")

    if not signal_path:
        print("ERROR: No signal DB found. Use --signal-db.", file=sys.stderr)
        return 1

    print("=" * 50)
    print("  Ontology-Enriched TDA Demo")
    print("=" * 50)
    print(f"\n  Signal DB:  {Path(signal_path).name}")
    print(f"  Corpus DB:  {Path(corpus_path).name if corpus_path else '(none)'}")
    print(f"  WordNet:    {Path(wordnet_dir).name if wordnet_dir else '(none)'}")

    # ── Run 1: Flat signal distance (baseline) ──────────────────────
    print("\n  Loading graph (flat signal distance)...")
    graph_flat = SignalDBGraph(signal_path)
    result_flat = run_pipeline(graph_flat, "RUN 1: Flat Signal Jaccard Distance")

    # ── Run 2: Ontology-enriched distance ───────────────────────────
    if not corpus_path and not wordnet_dir:
        print("\n  Skipping ontology run (no corpus DB or WordNet dir)")
        return 0

    print("\n  Loading graph (ontology-enriched distance)...")
    graph_ont = SignalDBGraph(signal_path)

    print("  Loading ontology...")
    t0 = time.time()
    ont = OntologyDistance(
        corpus_db_path=corpus_path,
        wordnet_dir=wordnet_dir,
    )
    ont_time = time.time() - t0

    stats = []
    if corpus_path:
        stats.append(f"{len(ont._ancestors)} concepts")
        stats.append(f"{sum(len(v) for v in ont._doc_concepts.values())} concept tags")
    if ont._sumo_index:
        stats.append(f"{len(ont._sumo_index):,} WordNet lemmas")
        stats.append(f"{len(ont._sumo_mappings):,} SUMO mappings")
    print(f"  Ontology loaded: {', '.join(stats)} ({ont_time:.1f}s)")

    # Wrap distance
    original = graph_ont.get_distance
    def enriched_distance(u, v, _ont=ont, _g=graph_ont):
        signals_u = _g._signals.get(u, set())
        signals_v = _g._signals.get(v, set())
        return _ont.combined_distance(u, v, signals_u, signals_v)
    graph_ont.get_distance = enriched_distance

    result_ont = run_pipeline(graph_ont, "RUN 2: Ontology-Enriched Distance (hierarchy + SUMO + signal)")

    # ── Comparison ──────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  COMPARISON: Flat vs Ontology-Enriched")
    print(f"{'═' * 50}")
    print(f"  {'Metric':<35} {'Flat':>8} {'Ontology':>10} {'Δ':>8}")
    print(f"  {'─' * 63}")

    comparisons = [
        ("Simplices", result_flat["simplices"], result_ont["simplices"]),
        ("Persistence intervals", result_flat["intervals"], result_ont["intervals"]),
        ("Pipeline time (s)", result_flat["time"], result_ont["time"]),
        ("Maturity score", result_flat["maturity"]["maturity_score"],
         result_ont["maturity"]["maturity_score"]),
        ("Connectedness", result_flat["maturity"]["connectedness"],
         result_ont["maturity"]["connectedness"]),
        ("Topological stability", result_flat["maturity"]["topological_stability"],
         result_ont["maturity"]["topological_stability"]),
        ("Dim 0 persistence entropy", result_flat["persist"]["dim_0_entropy"],
         result_ont["persist"]["dim_0_entropy"]),
        ("Dim 0 total persistence", result_flat["persist"]["dim_0_total_persistence"],
         result_ont["persist"]["dim_0_total_persistence"]),
        ("β₀ final", result_flat["betti"]["summary"]["betti_0_final"],
         result_ont["betti"]["summary"]["betti_0_final"]),
        ("β₁ final", result_flat["betti"]["summary"]["betti_1_final"],
         result_ont["betti"]["summary"]["betti_1_final"]),
    ]

    for label, flat_val, ont_val in comparisons:
        if isinstance(flat_val, float):
            delta = ont_val - flat_val
            print(f"  {label:<35} {flat_val:>8.4f} {ont_val:>10.4f} {delta:>+8.4f}")
        else:
            delta = ont_val - flat_val
            print(f"  {label:<35} {flat_val:>8} {ont_val:>10} {delta:>+8}")

    # ── Sample distance pairs ───────────────────────────────────────
    sample_distance_pairs(graph_flat, graph_ont, n_pairs=15)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
