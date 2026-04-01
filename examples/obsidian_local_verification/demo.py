#!/usr/bin/env python3
"""Obsidian-Local TDA Verification: hub pages, uplinks, GLN, and Zettelkasten topology.

Analyzes the obsidian-local corpus through four topological lenses:
1. Full corpus topology (all 1,402 docs via signal DB)
2. Hub-centric topology (curated directory pages with high edge count)
3. Uplink hierarchy topology (parent-child document relationships)
4. GLN proximity (Generalized Luhmann Numbers from obsidian-refs.json)

Hub pages (focus/foci pages) are human-curated directories — their internal
links are strong topological signals. The uplink hierarchy creates a DAG
of knowledge threads. GLN labels encode human-assessed topological closeness.

When the vault is available at --vault-path, also loads:
- obsidian-refs.json: 482 GLN refs, 1,922 wiki links, 557 uplinks
- EXTRACTION-SUMMARY.md: corpus analysis crib
- GLN-Sprint-26033-map.md: human-curated topological map of sprint 26033

Usage:
    cd pn-tda
    python3 examples/obsidian_local_verification/demo.py
    python3 examples/obsidian_local_verification/demo.py --vault-path ~/environment/obsidian-local
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
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

DEFAULT_SIGNAL_PATHS = [
    os.path.expanduser("~/azure-wiki-analysis/output/obsidian-signals.db"),
]
DEFAULT_CORPUS_PATHS = [
    os.path.expanduser(
        "~/pn-monorepo/pn-monorepo/projects/kitsap-searchengine-lite/data/pn-daynotes.db"
    ),
]
DEFAULT_WORDNET_PATHS = [
    os.path.expanduser("~/pn-monorepo/pn-monorepo/data/Ontologies/wordnet-mappings"),
]
DEFAULT_VAULT_PATHS = [
    os.path.expanduser("~/environment/obsidian-local"),
]

GLN_RE = re.compile(r'\b(\d+[xhk][0-9a-f]+[0-9a-z]*)\b', re.IGNORECASE)


def find_path(candidates, label):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_vault_artifacts(vault_path: str) -> dict:
    """Load Open Planter artifacts from the vault root."""
    artifacts = {}
    root = Path(vault_path)

    # obsidian-refs.json — cross-references
    refs_path = root / "obsidian-refs.json"
    if refs_path.is_file():
        with open(refs_path) as f:
            artifacts["obsidian_refs"] = json.load(f)

    # monorepo-refs.json — monorepo concept references
    mono_path = root / "monorepo-refs.json"
    if mono_path.is_file():
        with open(mono_path) as f:
            artifacts["monorepo_refs"] = json.load(f)

    # documentation-trace.json — spec traceability
    trace_path = root / "documentation-trace.json"
    if trace_path.is_file():
        with open(trace_path) as f:
            artifacts["doc_trace"] = json.load(f)

    # STI-Curriculum-Data.json — curriculum structure
    sti_path = root / "STI-Curriculum-Data.json"
    if sti_path.is_file():
        with open(sti_path) as f:
            artifacts["sti_curriculum"] = json.load(f)

    return artifacts


def analyze_gln_topology(artifacts: dict):
    """Analyze GLN (Generalized Luhmann Numbers) as topological proximity labels."""
    refs = artifacts.get("obsidian_refs", {})
    gln_refs = refs.get("gln_refs", [])
    if not gln_refs:
        return

    print(f"\n{'═' * 56}")
    print(f"  GLN TOPOLOGY (Generalized Luhmann Numbers)")
    print(f"{'═' * 56}")

    # Build GLN → file mapping
    gln_to_files: dict[str, set[str]] = defaultdict(set)
    file_to_glns: dict[str, set[str]] = defaultdict(set)
    for ref in gln_refs:
        gln = ref.get("gln", "")
        fpath = ref.get("file", "")
        if gln and fpath:
            gln_to_files[gln].add(fpath)
            file_to_glns[fpath].add(gln)

    print(f"\n  GLN references: {len(gln_refs)} total, {len(gln_to_files)} unique GLNs")
    print(f"  Files with GLNs: {len(file_to_glns)}")

    # GLN prefix clusters (shared prefix = topological neighbors)
    prefix_clusters: dict[str, set[str]] = defaultdict(set)
    for gln in gln_to_files:
        # Extract root prefix (first 2-3 chars before 'x'/'h'/'k')
        match = re.match(r'(\d+[xhk])', gln, re.IGNORECASE)
        if match:
            prefix_clusters[match.group(1)].update(gln_to_files[gln])

    print(f"\n  GLN prefix clusters (shared root = topological neighborhood):")
    for prefix, files in sorted(prefix_clusters.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {prefix:<8} → {len(files)} files")

    # Wiki links and uplinks from refs
    wiki_links = refs.get("wiki_links", [])
    uplinks = refs.get("uplinks", [])
    print(f"\n  Cross-reference totals:")
    print(f"    Wiki links: {len(wiki_links)} ({len(set(l.get('link','') for l in wiki_links))} unique targets)")
    print(f"    Uplinks:    {len(uplinks)} ({len(set(u.get('target','') for u in uplinks))} unique targets)")

    # Monorepo integration density
    mono = artifacts.get("monorepo_refs", {})
    if mono:
        mono_refs = mono.get("monorepo_refs", mono.get("obsidian_refs", []))
        if isinstance(mono_refs, list):
            print(f"    Monorepo refs: {len(mono_refs)}")

    # Doc trace
    trace = artifacts.get("doc_trace", {})
    if trace:
        shared = trace.get("shared_identifiers", [])
        orphans = trace.get("orphaned_identifiers", [])
        print(f"    Doc trace: {len(shared)} shared identifiers, {len(orphans)} orphaned")


def analyze_vault_structure(artifacts: dict):
    """Summarize vault structure from Open Planter artifacts."""
    refs = artifacts.get("obsidian_refs", {})
    if not refs:
        return

    print(f"\n{'═' * 56}")
    print(f"  VAULT ARTIFACT SUMMARY")
    print(f"{'═' * 56}")

    for key, label in [
        ("gln_refs", "GLN references"),
        ("sku_refs", "SKU references"),
        ("bead_refs", "Bead references"),
        ("sprint_refs", "Sprint references"),
        ("monorepo_refs", "Monorepo refs"),
        ("wiki_links", "Wiki links"),
        ("uplinks", "Uplinks"),
    ]:
        items = refs.get(key, [])
        if items:
            unique_key = {
                "gln_refs": "gln", "sku_refs": "sku", "bead_refs": "bead",
                "sprint_refs": "sprint", "monorepo_refs": "term",
                "wiki_links": "link", "uplinks": "target",
            }.get(key, "")
            unique_count = len(set(r.get(unique_key, "") for r in items)) if unique_key else "?"
            print(f"  {label:<20} {len(items):>6} total  {unique_count:>6} unique")


def print_features(label, graph, max_dimension=2, num_scales=10):
    """Run pipeline and print features."""
    doc_ids = list(graph.nodes())
    edges = list(graph.edges())

    t0 = time.time()
    builder = GraphFiltrationBuilder(max_dimension=max_dimension)
    st = builder.build(graph)
    intervals = PersistentHomology().compute(st)
    elapsed = time.time() - t0

    betti = BettiNumberExtractor().extract(
        intervals, num_scales=num_scales, epsilon_max=1.0, max_dimension=max_dimension,
    )
    persist = PersistenceFeatureExtractor().extract(intervals, max_dimension=max_dimension)
    maturity = ThreadMaturityScorer().score(
        intervals, graph, num_scales=num_scales, epsilon_max=1.0,
    )

    print(f"\n{'─' * 56}")
    print(f"  {label}")
    print(f"{'─' * 56}")
    print(f"  Nodes: {len(doc_ids):,}  Edges: {len(edges):,}  "
          f"Simplices: {st.num_simplices():,}  ({elapsed:.2f}s)")
    print(f"  Intervals: {len(intervals):,}  "
          f"(H0: {persist['dim_0_count']}, "
          f"H1: {persist['dim_1_count']}, "
          f"H2: {persist.get('dim_2_count', 0)})")
    print()
    print(f"  β₀={betti['summary']['betti_0_final']} components  "
          f"β₁={betti['summary']['betti_1_final']} loops  "
          f"β₂={betti['summary']['betti_2_final']} voids")
    print()
    print(f"  Maturity: {maturity['maturity_score']:.4f}")
    print(f"    connectedness={maturity['connectedness']:.4f}  "
          f"stability={maturity['topological_stability']:.4f}")
    print(f"    plateau={maturity['persistence_plateau']:.4f}  "
          f"dim_shift={maturity['dimensional_shift']:.4f}")
    print()
    print(f"  Persistence (H0): total={persist['dim_0_total_persistence']:.2f}  "
          f"entropy={persist['dim_0_entropy']:.4f}")
    if persist['dim_1_count'] > 0:
        print(f"  Persistence (H1): total={persist['dim_1_total_persistence']:.4f}  "
              f"entropy={persist['dim_1_entropy']:.4f}  count={persist['dim_1_count']}")

    return {"betti": betti, "persist": persist, "maturity": maturity}


def analyze_hub_structure(conn: sqlite3.Connection):
    """Identify hub pages and their topological role."""
    print(f"\n{'═' * 56}")
    print(f"  HUB PAGE ANALYSIS")
    print(f"{'═' * 56}")

    # Docs with most internal edges (obsidian_wikilink + uplink)
    rows = conn.execute("""
        SELECT d.source_path, d.doc_id, d.is_focus_hub, COUNT(*) as edge_count
        FROM edges e JOIN documents d ON e.source_doc_id = d.doc_id
        WHERE e.edge_type IN ('obsidian_wikilink', 'uplink')
        GROUP BY e.source_doc_id
        ORDER BY edge_count DESC
        LIMIT 20
    """).fetchall()

    print(f"\n  Top 20 pages by internal link count:")
    print(f"  {'Edges':>6} {'Hub':>4} Path")
    print(f"  {'─' * 50}")
    for r in rows:
        hub = " ★" if r[2] else "  "
        path = r[0]
        if len(path) > 60:
            path = "..." + path[-57:]
        print(f"  {r[3]:>6} {hub:>4} {path}")

    # Uplink hierarchy depth
    print(f"\n  Uplink hierarchy (parent → children):")
    targets = conn.execute("""
        SELECT target_path, COUNT(*) as cnt
        FROM edges WHERE edge_type = 'uplink'
        GROUP BY target_path
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    for t in targets:
        target = t[0]
        if len(target) > 50:
            target = "..." + target[-47:]
        print(f"    {t[1]:>3} children → {target}")


def analyze_classification_clusters(conn: sqlite3.Connection):
    """Show how JDN/CEC classification refs cluster documents."""
    print(f"\n{'═' * 56}")
    print(f"  CLASSIFICATION SIGNAL CLUSTERS")
    print(f"{'═' * 56}")

    # JDN clusters
    rows = conn.execute("""
        SELECT signal_value, COUNT(DISTINCT doc_id) as doc_count
        FROM signals
        WHERE signal_type = 'classification_ref'
        AND signal_value LIKE 'JDN%'
        GROUP BY signal_value
        ORDER BY doc_count DESC
        LIMIT 10
    """).fetchall()

    print(f"\n  JDN classification clusters:")
    for r in rows:
        print(f"    {r[0]:<12} {r[1]} docs")

    # SUMO/MILO
    rows2 = conn.execute("""
        SELECT signal_value, COUNT(DISTINCT doc_id) as doc_count
        FROM signals
        WHERE signal_type = 'classification_ref'
        AND signal_value LIKE '%SUMO%' OR signal_value LIKE '%MILO%'
        GROUP BY signal_value
        ORDER BY doc_count DESC
    """).fetchall()
    if rows2:
        print(f"\n  SUMO/MILO clusters:")
        for r in rows2:
            print(f"    {r[0]:<12} {r[1]} docs")


def main():
    parser = argparse.ArgumentParser(description="Obsidian-Local TDA Verification")
    parser.add_argument("--signal-db", help="Path to obsidian-signals.db")
    parser.add_argument("--corpus-db", help="Path to corpus DB (for ontology hierarchy)")
    parser.add_argument("--wordnet-dir", help="Path to WordNetMappings")
    parser.add_argument("--vault-path", help="Path to obsidian-local vault root")
    parser.add_argument("--hub-threshold", type=int, default=10,
                        help="Min internal edges to count as hub (default: 10)")
    args = parser.parse_args()

    signal_path = args.signal_db or find_path(DEFAULT_SIGNAL_PATHS, "signal DB")
    corpus_path = args.corpus_db or find_path(DEFAULT_CORPUS_PATHS, "corpus DB")
    wordnet_dir = args.wordnet_dir or find_path(DEFAULT_WORDNET_PATHS, "WordNet")
    vault_path = args.vault_path or find_path(DEFAULT_VAULT_PATHS, "vault")

    if not signal_path:
        print("ERROR: No obsidian signal DB found. Use --signal-db.", file=sys.stderr)
        return 1

    print("=" * 56)
    print("  Obsidian-Local TDA Verification")
    print("=" * 56)
    print(f"\n  Signal DB: {Path(signal_path).name}")

    sig_conn = sqlite3.connect(signal_path)
    sig_conn.row_factory = sqlite3.Row
    doc_count = sig_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    edge_count = sig_conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    wikilink_count = sig_conn.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type = 'obsidian_wikilink'"
    ).fetchone()[0]
    uplink_count = sig_conn.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type = 'uplink'"
    ).fetchone()[0]
    signal_count = sig_conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print(f"  Docs: {doc_count:,}  Signals: {signal_count:,}")
    print(f"  Edges: {edge_count:,} (wikilinks: {wikilink_count:,}, uplinks: {uplink_count:,})")

    # ── Hub & Classification Analysis ─────────────────────────────
    analyze_hub_structure(sig_conn)
    analyze_classification_clusters(sig_conn)
    sig_conn.close()

    # ── Vault Artifacts (Open Planter) ────────────────────────────
    artifacts = {}
    if vault_path:
        print(f"\n  Vault: {vault_path}")
        artifacts = load_vault_artifacts(vault_path)
        if artifacts:
            analyze_vault_structure(artifacts)
            analyze_gln_topology(artifacts)
        else:
            print("  (no vault artifacts found)")
    else:
        print("\n  Vault: (not available — use --vault-path for GLN analysis)")

    # ── RUN 1: Full corpus topology ──────────────────────────────
    print(f"\n{'═' * 56}")
    print(f"  TOPOLOGICAL ANALYSIS")
    print(f"{'═' * 56}")

    graph = SignalDBGraph(signal_path)
    result_full = print_features("Full Corpus (all docs, flat signal distance)", graph)

    # ── RUN 2: With ontology enrichment ──────────────────────────
    if corpus_path or wordnet_dir:
        graph_ont = SignalDBGraph(signal_path)
        ont = OntologyDistance(
            corpus_db_path=corpus_path,
            wordnet_dir=wordnet_dir,
        )

        original = graph_ont.get_distance
        def enriched(u, v, _ont=ont, _g=graph_ont):
            return _ont.combined_distance(
                u, v, _g._signals.get(u, set()), _g._signals.get(v, set()),
            )
        graph_ont.get_distance = enriched

        result_ont = print_features(
            "Full Corpus (ontology-enriched: hierarchy + SUMO + signal)", graph_ont,
        )

        # Compare
        print(f"\n{'═' * 56}")
        print(f"  COMPARISON: Flat vs Ontology")
        print(f"{'═' * 56}")
        print(f"  {'Metric':<30} {'Flat':>8} {'Ontology':>10}")
        print(f"  {'─' * 50}")
        pairs = [
            ("Maturity", result_full["maturity"]["maturity_score"],
             result_ont["maturity"]["maturity_score"]),
            ("Connectedness", result_full["maturity"]["connectedness"],
             result_ont["maturity"]["connectedness"]),
            ("H0 total persistence",
             result_full["persist"]["dim_0_total_persistence"],
             result_ont["persist"]["dim_0_total_persistence"]),
            ("H0 entropy", result_full["persist"]["dim_0_entropy"],
             result_ont["persist"]["dim_0_entropy"]),
            ("H1 count", result_full["persist"]["dim_1_count"],
             result_ont["persist"]["dim_1_count"]),
        ]
        for label, flat, ont_v in pairs:
            if isinstance(flat, float):
                print(f"  {label:<30} {flat:>8.4f} {ont_v:>10.4f}")
            else:
                print(f"  {label:<30} {flat:>8} {ont_v:>10}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
