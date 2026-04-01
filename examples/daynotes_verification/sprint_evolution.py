#!/usr/bin/env python3
"""Daynotes Sprint Evolution: TDA metrics across 170 sprints (2010-2025).

The Daynotes wiki is a long-running Azure DevOps wiki spanning 15 years and
5,389 documents. Unlike obsidian-local, heading content is read from the
corpus DB's raw_content column (vault not needed on this host).

Three topology layers computed per sprint:
1. Graph filtration (inter-document edges: wiki_links + uplinks)
2. Vietoris-Rips (all-pairs signal Jaccard — for comparison)
3. Heading topology (intra-document outline structure from raw_content)

Usage:
    cd pn-tda
    python3 examples/daynotes_verification/sprint_evolution.py
    python3 examples/daynotes_verification/sprint_evolution.py --cumulative
    python3 examples/daynotes_verification/sprint_evolution.py --recent 20
    python3 examples/daynotes_verification/sprint_evolution.py --tsv > results.tsv
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
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
from pn_tda.utils.heading_graph import extract_heading_edges, heading_depth_stats

DEFAULT_SIGNAL_PATHS = [
    os.path.expanduser("~/azure-wiki-analysis/output/signals.db"),
]
DEFAULT_CORPUS_PATHS = [
    os.path.expanduser(
        "~/pn-monorepo/pn-monorepo/projects/kitsap-searchengine-lite/data/pn-daynotes.db"
    ),
]


def find_path(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def build_sprint_db(source_conn, sprint_ids, db_path):
    """Create signal DB subset for given sprints."""
    dst = sqlite3.connect(db_path)
    for row in source_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%'"
    ):
        dst.execute(row[0])

    ph = ",".join("?" for _ in sprint_ids)
    doc_rows = source_conn.execute(
        f"SELECT * FROM documents WHERE sprint_id IN ({ph})", sprint_ids
    ).fetchall()
    doc_ids = {r[0] for r in doc_rows}
    if not doc_ids:
        dst.close()
        return db_path

    id_ph = ",".join("?" for _ in doc_ids)
    id_list = list(doc_ids)

    for row in doc_rows:
        dst.execute(f"INSERT OR IGNORE INTO documents VALUES ({','.join('?' for _ in row)})", tuple(row))
    for row in source_conn.execute(f"SELECT * FROM signals WHERE doc_id IN ({id_ph})", id_list):
        dst.execute(f"INSERT OR IGNORE INTO signals VALUES ({','.join('?' for _ in row)})", tuple(row))
    for row in source_conn.execute(f"SELECT * FROM edges WHERE source_doc_id IN ({id_ph})", id_list):
        dst.execute(f"INSERT OR IGNORE INTO edges VALUES ({','.join('?' for _ in row)})", tuple(row))

    dst.commit()
    dst.close()
    return db_path


def _extract(graph, st, intervals):
    betti = BettiNumberExtractor().extract(intervals, num_scales=10, epsilon_max=1.0, max_dimension=2)
    persist = PersistenceFeatureExtractor().extract(intervals, max_dimension=2)
    maturity = ThreadMaturityScorer().score(intervals, graph, num_scales=10, epsilon_max=1.0)
    return {
        "simplices": st.num_simplices(), "b0": betti["summary"]["betti_0_final"],
        "b1": betti["summary"]["betti_1_final"], "h0_entropy": persist["dim_0_entropy"],
        "h1_count": persist["dim_1_count"], "maturity": maturity["maturity_score"],
        "connected": maturity["connectedness"], "stability": maturity["topological_stability"],
        "dim_shift": maturity["dimensional_shift"],
    }


def compute_metrics(db_path):
    """Run Graph + VR pipelines."""
    graph = SignalDBGraph(db_path)
    doc_ids = list(graph.nodes())
    edges = list(graph.edges())
    if len(doc_ids) < 2:
        return None

    gf_st = GraphFiltrationBuilder(max_dimension=2).build(graph)
    gf_iv = PersistentHomology().compute(gf_st)
    gf = _extract(graph, gf_st, gf_iv)

    vr_st = VietorisRipsBuilder(epsilon_max=1.0, max_dimension=2).build(graph)
    vr_iv = PersistentHomology().compute(vr_st)
    vr = _extract(graph, vr_st, vr_iv)

    return {"nodes": len(doc_ids), "edges": len(edges), "gf": gf, "vr": vr}


def compute_heading_metrics_from_db(corpus_conn, source_paths):
    """Compute heading topology from corpus DB raw_content (no vault needed)."""
    if not source_paths:
        return None

    all_nodes = []
    all_edges = []
    total_headings = 0
    max_depth = 0
    branching_sum = 0.0
    branching_count = 0
    docs_with_headings = 0

    ph = ",".join("?" for _ in source_paths)
    rows = corpus_conn.execute(
        f"SELECT source_path, raw_content FROM documents WHERE source_path IN ({ph})",
        source_paths,
    ).fetchall()

    for row in rows:
        content = row[1] or ""
        if not content:
            continue

        stats = heading_depth_stats(content)
        if stats["heading_count"] == 0:
            continue

        docs_with_headings += 1
        total_headings += stats["heading_count"]
        max_depth = max(max_depth, stats["max_depth"])
        if stats["branching_factor"] > 0:
            branching_sum += stats["branching_factor"]
            branching_count += 1

        doc_id = row[0]
        nodes, edges = extract_heading_edges(doc_id, content)
        all_nodes.extend(nodes)
        for e in edges:
            all_edges.append((e.source, e.target, e.filtration))

    if not all_edges:
        return {
            "docs": docs_with_headings, "headings": total_headings,
            "depth": max_depth, "branch": 0.0, "hb0": 0, "hb1": 0,
        }

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
        "docs": docs_with_headings, "headings": total_headings,
        "depth": max_depth,
        "branch": branching_sum / branching_count if branching_count else 0.0,
        "hb0": betti.get(0, 0), "hb1": betti.get(1, 0),
    }


def sprint_label(sid):
    if len(sid) != 5:
        return sid
    yy, mm, s = sid[:2], sid[2:4], sid[4]
    months = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May",
              "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct",
              "11": "Nov", "12": "Dec"}
    return f"20{yy}-{months.get(mm, mm)}-{s}"


def main():
    parser = argparse.ArgumentParser(description="Daynotes Sprint Evolution")
    parser.add_argument("--signal-db", help="Path to Daynotes signals.db")
    parser.add_argument("--corpus-db", help="Path to pn-daynotes.db (for heading content)")
    parser.add_argument("--cumulative", action="store_true")
    parser.add_argument("--recent", type=int, default=0,
                        help="Only show the N most recent sprints (default: all)")
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args()

    signal_path = args.signal_db or find_path(DEFAULT_SIGNAL_PATHS)
    corpus_path = args.corpus_db or find_path(DEFAULT_CORPUS_PATHS)

    if not signal_path:
        print("ERROR: No signal DB found.", file=sys.stderr)
        return 1

    source_conn = sqlite3.connect(signal_path)
    source_conn.row_factory = sqlite3.Row

    sprints = source_conn.execute(
        "SELECT sprint_id, COUNT(*) as cnt FROM documents "
        "WHERE sprint_id IS NOT NULL AND length(sprint_id) = 5 "
        "GROUP BY sprint_id ORDER BY sprint_id"
    ).fetchall()

    if args.recent > 0:
        sprints = sprints[-args.recent:]

    # Build sprint → source_paths map for heading analysis
    corpus_conn = None
    sprint_paths: dict[str, list[str]] = {}
    has_headings = False
    if corpus_path:
        corpus_conn = sqlite3.connect(corpus_path)
        corpus_conn.row_factory = sqlite3.Row
        # Map signal DB source_paths to corpus DB source_paths via filename
        sig_paths = source_conn.execute(
            "SELECT sprint_id, source_path FROM documents WHERE sprint_id IS NOT NULL"
        ).fetchall()
        # Build filename → corpus source_path index
        corpus_paths_by_name = {}
        for row in corpus_conn.execute("SELECT source_path FROM documents"):
            sp = row[0]
            name = sp.replace("\\", "/").split("/")[-1]
            corpus_paths_by_name[name] = sp

        for row in sig_paths:
            sid = row["sprint_id"]
            sp = row["source_path"]
            name = sp.replace("\\", "/").split("/")[-1]
            corpus_sp = corpus_paths_by_name.get(name)
            if corpus_sp:
                if sid not in sprint_paths:
                    sprint_paths[sid] = []
                sprint_paths[sid].append(corpus_sp)
                has_headings = True

    mode = "cumulative" if args.cumulative else "per-sprint"
    print(f"{'═' * 115}")
    print(f"  Daynotes Sprint Evolution ({mode})")
    print(f"{'═' * 115}")
    print(f"  Signal DB: {Path(signal_path).name} ({len(sprints)} sprints)")
    if corpus_path:
        print(f"  Corpus DB: {Path(corpus_path).name} (heading content)")
    print()

    hdg_hdr = " │ Heading Topology" if has_headings else ""
    hdg_col = f" │ {'Hdg':>5} {'Dp':>3} {'hβ₀':>5} {'hβ₁':>5} {'Br':>4}" if has_headings else ""
    if not args.tsv:
        print(f"  {'':22}{'':10}│ {'Graph Filtration':^29} │ {'Vietoris-Rips':^29}{hdg_hdr}")
        print(f"  {'Sprint':<7} {'Label':<11} {'N':>4} {'E':>4}"
              f"│ {'Spl':>6} {'β₀':>4} {'β₁':>4} {'Mat':>5} {'Con':>5}"
              f" │ {'Spl':>6} {'β₀':>4} {'β₁':>4} {'Mat':>5} {'Con':>5}"
              f"{hdg_col}")
        print(f"  {'─' * (92 + (27 if has_headings else 0))}")

    results = []
    cumulative_sprints = []
    cumulative_paths: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for sprint_id, doc_count in sprints:
            if doc_count < 2:
                continue

            if args.cumulative:
                cumulative_sprints.append(sprint_id)
                slist = cumulative_sprints
                if sprint_id in sprint_paths:
                    cumulative_paths.extend(sprint_paths[sprint_id])
            else:
                slist = [sprint_id]

            db_path = os.path.join(tmpdir, f"sprint_{sprint_id}.db")
            build_sprint_db(source_conn, slist, db_path)
            metrics = compute_metrics(db_path)
            os.unlink(db_path)

            if metrics is None:
                continue

            # Heading metrics
            hdg = None
            if has_headings and corpus_conn:
                paths = cumulative_paths if args.cumulative else sprint_paths.get(sprint_id, [])
                if paths:
                    hdg = compute_heading_metrics_from_db(corpus_conn, paths)
                    metrics["hdg"] = hdg

            label = sprint_label(sprint_id)
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
                    vals += [str(hdg["headings"]), str(hdg["depth"]),
                             str(hdg["hb0"]), str(hdg["hb1"]), f"{hdg['branch']:.1f}"]
                print("\t".join(vals))
            else:
                hdg_str = ""
                if hdg:
                    hdg_str = (f" │ {hdg['headings']:>5} {hdg['depth']:>3} "
                               f"{hdg['hb0']:>5} {hdg['hb1']:>5} {hdg['branch']:>4.1f}")
                print(
                    f"  {sprint_id:<7} {label:<11} {metrics['nodes']:>4} "
                    f"{metrics['edges']:>4}"
                    f"│ {gf['simplices']:>6} {gf['b0']:>4} {gf['b1']:>4} "
                    f"{gf['maturity']:>5.2f} {gf['connected']:>5.2f}"
                    f" │ {vr['simplices']:>6} {vr['b0']:>4} {vr['b1']:>4} "
                    f"{vr['maturity']:>5.2f} {vr['connected']:>5.2f}"
                    f"{hdg_str}"
                )

    source_conn.close()
    if corpus_conn:
        corpus_conn.close()

    if not args.tsv and len(results) >= 2:
        first, last = results[0], results[-1]
        print(f"\n  {'─' * (92 + (27 if has_headings else 0))}")
        print(f"  TREND ({first['sprint']} → {last['sprint']}, {len(results)} sprints):")
        for builder, key in [("Graph", "gf"), ("VR   ", "vr")]:
            print(f"    {builder}: maturity {first[key]['maturity']:.3f}→{last[key]['maturity']:.3f}  "
                  f"conn {first[key]['connected']:.3f}→{last[key]['connected']:.3f}")

        gf_peak = max(results, key=lambda r: r["gf"]["maturity"])
        print(f"    Peak maturity: {gf_peak['sprint']} ({gf_peak['label']}) = {gf_peak['gf']['maturity']:.3f}")

        if has_headings:
            hdg_results = [r for r in results if "hdg" in r and r["hdg"]]
            if hdg_results:
                peak_hdg = max(hdg_results, key=lambda r: r["hdg"]["headings"])
                peak_loops = max(hdg_results, key=lambda r: r["hdg"]["hb1"])
                print(f"    Most headings: {peak_hdg['sprint']} ({peak_hdg['hdg']['headings']})")
                print(f"    Most heading loops: {peak_loops['sprint']} (hβ₁={peak_loops['hdg']['hb1']})")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
