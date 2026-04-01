#!/usr/bin/env python3
"""Sprint Evolution: TDA metrics across sequential sprints.

Processes each sprint from the obsidian-local corpus sequentially,
computing topological features per sprint and presenting a tabular
result showing how the knowledge graph evolves over time.

Sprints are identified by the pattern Fleeting/FY26xxQy/<5-digit-sprint>/
and processed in chronological order.

Usage:
    cd pn-tda
    python3 examples/obsidian_local_verification/sprint_evolution.py
    python3 examples/obsidian_local_verification/sprint_evolution.py --cumulative
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import GraphFiltrationBuilder, VietorisRipsBuilder
from pn_tda.core.persistence import PersistentHomology, betti_numbers
from pn_tda.core.simplex_tree import SimplexTree
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor
from pn_tda.utils.heading_graph import (
    extract_heading_edges,
    heading_depth_stats,
)

DEFAULT_SIGNAL_PATHS = [
    os.path.expanduser("~/azure-wiki-analysis/output/obsidian-signals.db"),
]
DEFAULT_VAULT_PATHS = [
    os.path.expanduser("~/environment/obsidian-local"),
]


def find_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def compute_heading_metrics(
    vault_path: str, source_paths: list[str],
) -> dict:
    """Compute heading-level TDA metrics for a set of documents.

    Reads markdown files from the vault, extracts heading trees,
    builds a combined heading graph, and runs TDA on it.
    """
    all_nodes: list[str] = []
    all_edges: list[tuple[str, str, float]] = []  # (src, tgt, filtration)
    total_headings = 0
    total_docs_with_headings = 0
    max_depth_seen = 0
    total_branching = 0.0
    branching_count = 0

    vault = Path(vault_path)

    for sp in source_paths:
        # source_path in DB may not match vault layout exactly
        # Try direct match and common prefixes
        candidates = [vault / sp, vault / sp.lstrip("/")]
        md_path = None
        for c in candidates:
            if c.is_file():
                md_path = c
                break
        if md_path is None:
            continue

        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        stats = heading_depth_stats(content)
        if stats["heading_count"] == 0:
            continue

        total_docs_with_headings += 1
        total_headings += stats["heading_count"]
        max_depth_seen = max(max_depth_seen, stats["max_depth"])
        if stats["branching_factor"] > 0:
            total_branching += stats["branching_factor"]
            branching_count += 1

        doc_id = sp  # use source_path as doc_id for heading graph
        nodes, edges = extract_heading_edges(doc_id, content)
        all_nodes.extend(nodes)
        for e in edges:
            all_edges.append((e.source, e.target, e.filtration))

    if not all_edges:
        return {
            "docs_with_headings": total_docs_with_headings,
            "total_headings": total_headings,
            "max_depth": max_depth_seen,
            "avg_branching": 0.0,
            "heading_nodes": len(all_nodes),
            "heading_edges": 0,
            "heading_b0": 0,
            "heading_b1": 0,
        }

    # Build simplex tree from heading edges
    node_to_idx = {}
    idx = 0
    for n in all_nodes:
        if n not in node_to_idx:
            node_to_idx[n] = idx
            idx += 1

    st = SimplexTree()
    for i in range(idx):
        st.insert((i,), 0.0)
    for src, tgt, filt in all_edges:
        if src in node_to_idx and tgt in node_to_idx:
            si, ti = node_to_idx[src], node_to_idx[tgt]
            if si != ti:
                st.insert((min(si, ti), max(si, ti)), filt)

    intervals = PersistentHomology().compute(st)
    betti = betti_numbers(intervals, at_scale=0.99)

    return {
        "docs_with_headings": total_docs_with_headings,
        "total_headings": total_headings,
        "max_depth": max_depth_seen,
        "avg_branching": total_branching / branching_count if branching_count else 0.0,
        "heading_nodes": idx,
        "heading_edges": len(all_edges),
        "heading_b0": betti.get(0, 0),
        "heading_b1": betti.get(1, 0),
    }


def build_sprint_db(
    source_conn: sqlite3.Connection,
    sprint_ids: list[str],
    db_path: str,
) -> str:
    """Create a signal DB containing only documents from the given sprints."""
    dst = sqlite3.connect(db_path)

    # Copy schema (skip internal tables)
    for row in source_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%'"
    ):
        dst.execute(row[0])

    # Get doc_ids for these sprints
    placeholders = ",".join("?" for _ in sprint_ids)
    doc_rows = source_conn.execute(
        f"SELECT * FROM documents WHERE sprint_id IN ({placeholders})",
        sprint_ids,
    ).fetchall()
    doc_ids = {r[0] for r in doc_rows}  # doc_id is first column

    if not doc_ids:
        dst.close()
        return db_path

    id_ph = ",".join("?" for _ in doc_ids)
    id_list = list(doc_ids)

    for row in doc_rows:
        dst.execute(
            f"INSERT OR IGNORE INTO documents VALUES ({','.join('?' for _ in row)})",
            tuple(row),
        )

    for row in source_conn.execute(
        f"SELECT * FROM signals WHERE doc_id IN ({id_ph})", id_list
    ):
        dst.execute(
            f"INSERT OR IGNORE INTO signals VALUES ({','.join('?' for _ in row)})",
            tuple(row),
        )

    for row in source_conn.execute(
        f"SELECT * FROM edges WHERE source_doc_id IN ({id_ph})", id_list
    ):
        dst.execute(
            f"INSERT OR IGNORE INTO edges VALUES ({','.join('?' for _ in row)})",
            tuple(row),
        )

    dst.commit()
    dst.close()
    return db_path


def _extract_features(graph, st, intervals):
    """Extract feature dict from a built simplex tree + intervals."""
    betti = BettiNumberExtractor().extract(
        intervals, num_scales=10, epsilon_max=1.0, max_dimension=2,
    )
    persist = PersistenceFeatureExtractor().extract(intervals, max_dimension=2)
    maturity = ThreadMaturityScorer().score(
        intervals, graph, num_scales=10, epsilon_max=1.0,
    )
    return {
        "simplices": st.num_simplices(),
        "intervals": len(intervals),
        "b0": betti["summary"]["betti_0_final"],
        "b1": betti["summary"]["betti_1_final"],
        "h0_entropy": persist["dim_0_entropy"],
        "h1_count": persist["dim_1_count"],
        "maturity": maturity["maturity_score"],
        "connected": maturity["connectedness"],
        "stability": maturity["topological_stability"],
        "dim_shift": maturity["dimensional_shift"],
    }


def compute_sprint_metrics(db_path: str, epsilon_max: float = 1.0) -> dict | None:
    """Run both Graph and VR pipelines on a sprint's signal DB."""
    graph = SignalDBGraph(db_path)
    doc_ids = list(graph.nodes())
    edges = list(graph.edges())

    if len(doc_ids) < 2:
        return None

    # Graph filtration (O(|V|+|E|))
    gf_builder = GraphFiltrationBuilder(max_dimension=2)
    gf_st = gf_builder.build(graph)
    gf_intervals = PersistentHomology().compute(gf_st)
    gf = _extract_features(graph, gf_st, gf_intervals)

    # Vietoris-Rips (O(n²)) — feasible since per-sprint n ≤ 88
    vr_builder = VietorisRipsBuilder(epsilon_max=epsilon_max, max_dimension=2)
    vr_st = vr_builder.build(graph)
    vr_intervals = PersistentHomology().compute(vr_st)
    vr = _extract_features(graph, vr_st, vr_intervals)

    return {
        "nodes": len(doc_ids),
        "edges": len(edges),
        "gf": gf,
        "vr": vr,
    }


def sprint_to_label(sprint_id: str) -> str:
    """Convert sprint ID to human-readable label."""
    # Pattern: YYMMS where YY=year, MM=month, S=sprint-in-month
    if len(sprint_id) != 5:
        return sprint_id
    yy = sprint_id[:2]
    mm = sprint_id[2:4]
    s = sprint_id[4]
    months = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
    }
    return f"20{yy}-{months.get(mm, mm)}-{s}"


def main():
    parser = argparse.ArgumentParser(
        description="Sprint Evolution: TDA metrics across sequential sprints"
    )
    parser.add_argument("--signal-db", help="Path to obsidian-signals.db")
    parser.add_argument("--vault-path", help="Path to obsidian-local vault (for heading analysis)")
    parser.add_argument("--cumulative", action="store_true",
                        help="Accumulate docs across sprints (growing corpus)")
    parser.add_argument("--tsv", action="store_true",
                        help="Output TSV instead of formatted table")
    args = parser.parse_args()

    signal_path = args.signal_db or find_path(DEFAULT_SIGNAL_PATHS)
    if not signal_path:
        print("ERROR: No signal DB found. Use --signal-db.", file=sys.stderr)
        return 1

    vault_path = args.vault_path or find_path(DEFAULT_VAULT_PATHS)

    source_conn = sqlite3.connect(signal_path)
    source_conn.row_factory = sqlite3.Row

    # Get all sprints in order
    sprints = source_conn.execute(
        "SELECT sprint_id, COUNT(*) as cnt FROM documents "
        "WHERE sprint_id IS NOT NULL "
        "GROUP BY sprint_id ORDER BY sprint_id"
    ).fetchall()

    mode = "cumulative" if args.cumulative else "per-sprint"
    has_vault = vault_path is not None
    print(f"{'═' * 110}")
    print(f"  Sprint Evolution — Obsidian-Local TDA ({mode})")
    print(f"{'═' * 110}")
    print(f"  Signal DB: {Path(signal_path).name}")
    if has_vault:
        print(f"  Vault:     {vault_path}")
    print(f"  Sprints: {len(sprints)}  Mode: {mode}\n")

    # Collect source_paths per sprint for heading analysis
    sprint_paths: dict[str, list[str]] = {}
    if has_vault:
        for row in source_conn.execute(
            "SELECT sprint_id, source_path FROM documents WHERE sprint_id IS NOT NULL"
        ):
            sid = row["sprint_id"]
            if sid not in sprint_paths:
                sprint_paths[sid] = []
            sprint_paths[sid].append(row["source_path"])

    # Header
    hdr_hdg = " │ Headings" if has_vault else ""
    hdr_cols = " │ {'Hdg':>4} {'Dp':>3} {'hβ₀':>4} {'hβ₁':>4}" if has_vault else ""
    if args.tsv:
        cols = [
            "sprint", "label", "docs", "edges",
            "gf_simpl", "gf_β₀", "gf_β₁", "gf_matur", "gf_conn",
            "vr_simpl", "vr_β₀", "vr_β₁", "vr_matur", "vr_conn",
        ]
        if has_vault:
            cols += ["hdg_count", "hdg_depth", "hdg_β₀", "hdg_β₁", "hdg_branch"]
        print("\t".join(cols))
    else:
        vault_hdr = " │ {'Heading Topology':^22}" if has_vault else ""
        vault_col = f" │ {'Hdg':>4} {'Dp':>3} {'hβ₀':>4} {'hβ₁':>4} {'Br':>4}" if has_vault else ""
        print(f"  {'':21} {'':11}│ {'Graph Filtration':^29} │ {'Vietoris-Rips':^29}{vault_hdr}")
        print(f"  {'Sprint':<7} {'Label':<10} {'N':>4} {'E':>4}"
              f"│ {'Spl':>6} {'β₀':>4} {'β₁':>4} {'Mat':>5} {'Con':>5}"
              f" │ {'Spl':>6} {'β₀':>4} {'β₁':>4} {'Mat':>5} {'Con':>5}"
              f"{vault_col}")
        print(f"  {'─' * (90 + (24 if has_vault else 0))}")

    results = []
    cumulative_sprints = []
    cumulative_paths: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for sprint_id, doc_count in sprints:
            if doc_count < 2:
                continue

            if args.cumulative:
                cumulative_sprints.append(sprint_id)
                sprint_list = cumulative_sprints
                if sprint_id in sprint_paths:
                    cumulative_paths.extend(sprint_paths[sprint_id])
            else:
                sprint_list = [sprint_id]

            db_path = os.path.join(tmpdir, f"sprint_{sprint_id}.db")
            build_sprint_db(source_conn, sprint_list, db_path)
            metrics = compute_sprint_metrics(db_path)
            os.unlink(db_path)

            if metrics is None:
                continue

            # Heading metrics
            hdg = None
            if has_vault:
                paths = cumulative_paths if args.cumulative else sprint_paths.get(sprint_id, [])
                if paths:
                    hdg = compute_heading_metrics(vault_path, paths)
                    metrics["hdg"] = hdg

            label = sprint_to_label(sprint_id)
            metrics["sprint"] = sprint_id
            metrics["label"] = label
            results.append(metrics)

            gf = metrics["gf"]
            vr = metrics["vr"]

            if args.tsv:
                vals = [
                    sprint_id, label, str(metrics["nodes"]), str(metrics["edges"]),
                    str(gf["simplices"]), str(gf["b0"]), str(gf["b1"]),
                    f"{gf['maturity']:.4f}", f"{gf['connected']:.4f}",
                    str(vr["simplices"]), str(vr["b0"]), str(vr["b1"]),
                    f"{vr['maturity']:.4f}", f"{vr['connected']:.4f}",
                ]
                if hdg:
                    vals += [
                        str(hdg["total_headings"]), str(hdg["max_depth"]),
                        str(hdg["heading_b0"]), str(hdg["heading_b1"]),
                        f"{hdg['avg_branching']:.1f}",
                    ]
                print("\t".join(vals))
            else:
                hdg_str = ""
                if hdg:
                    hdg_str = (f" │ {hdg['total_headings']:>4} {hdg['max_depth']:>3} "
                               f"{hdg['heading_b0']:>4} {hdg['heading_b1']:>4} "
                               f"{hdg['avg_branching']:>4.1f}")
                print(
                    f"  {sprint_id:<7} {label:<10} {metrics['nodes']:>4} "
                    f"{metrics['edges']:>4}"
                    f"│ {gf['simplices']:>6} {gf['b0']:>4} {gf['b1']:>4} "
                    f"{gf['maturity']:>5.2f} {gf['connected']:>5.2f}"
                    f" │ {vr['simplices']:>6} {vr['b0']:>4} {vr['b1']:>4} "
                    f"{vr['maturity']:>5.2f} {vr['connected']:>5.2f}"
                    f"{hdg_str}"
                )

    source_conn.close()

    if not args.tsv and len(results) >= 2:
        first = results[0]
        last = results[-1]
        print(f"\n  {'─' * 90}")
        print(f"  TREND ({first['sprint']} → {last['sprint']}):")
        for builder, key in [("Graph", "gf"), ("VR   ", "vr")]:
            f_m = first[key]["maturity"]
            l_m = last[key]["maturity"]
            f_c = first[key]["connected"]
            l_c = last[key]["connected"]
            print(f"    {builder}: maturity {f_m:.3f}→{l_m:.3f} (Δ{l_m-f_m:+.3f})  "
                  f"conn {f_c:.3f}→{l_c:.3f} (Δ{l_c-f_c:+.3f})")

        # Peaks
        gf_peak = max(results, key=lambda r: r["gf"]["maturity"])
        vr_peak = max(results, key=lambda r: r["vr"]["maturity"])
        print(f"    Peak maturity: Graph={gf_peak['sprint']} ({gf_peak['gf']['maturity']:.3f})  "
              f"VR={vr_peak['sprint']} ({vr_peak['vr']['maturity']:.3f})")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
