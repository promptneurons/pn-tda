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
from pn_tda.core.filtration import GraphFiltrationBuilder
from pn_tda.core.persistence import PersistentHomology, betti_numbers
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor

DEFAULT_SIGNAL_PATHS = [
    os.path.expanduser("~/azure-wiki-analysis/output/obsidian-signals.db"),
]


def find_path(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


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


def compute_sprint_metrics(db_path: str) -> dict | None:
    """Run TDA pipeline on a sprint's signal DB and return metrics."""
    graph = SignalDBGraph(db_path)
    doc_ids = list(graph.nodes())
    edges = list(graph.edges())

    if len(doc_ids) < 2:
        return None

    builder = GraphFiltrationBuilder(max_dimension=2)
    st = builder.build(graph)
    intervals = PersistentHomology().compute(st)

    betti = BettiNumberExtractor().extract(
        intervals, num_scales=10, epsilon_max=1.0, max_dimension=2,
    )
    persist = PersistenceFeatureExtractor().extract(intervals, max_dimension=2)
    maturity = ThreadMaturityScorer().score(
        intervals, graph, num_scales=10, epsilon_max=1.0,
    )

    return {
        "nodes": len(doc_ids),
        "edges": len(edges),
        "simplices": st.num_simplices(),
        "b0": betti["summary"]["betti_0_final"],
        "b1": betti["summary"]["betti_1_final"],
        "b0_max": betti["summary"]["betti_0_max"],
        "h0_count": persist["dim_0_count"],
        "h0_total": persist["dim_0_total_persistence"],
        "h0_entropy": persist["dim_0_entropy"],
        "h1_count": persist["dim_1_count"],
        "maturity": maturity["maturity_score"],
        "connected": maturity["connectedness"],
        "stability": maturity["topological_stability"],
        "plateau": maturity["persistence_plateau"],
        "dim_shift": maturity["dimensional_shift"],
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
    parser.add_argument("--cumulative", action="store_true",
                        help="Accumulate docs across sprints (growing corpus)")
    parser.add_argument("--tsv", action="store_true",
                        help="Output TSV instead of formatted table")
    args = parser.parse_args()

    signal_path = args.signal_db or find_path(DEFAULT_SIGNAL_PATHS)
    if not signal_path:
        print("ERROR: No signal DB found. Use --signal-db.", file=sys.stderr)
        return 1

    source_conn = sqlite3.connect(signal_path)
    source_conn.row_factory = sqlite3.Row

    # Get all sprints in order
    sprints = source_conn.execute(
        "SELECT sprint_id, COUNT(*) as cnt FROM documents "
        "WHERE sprint_id IS NOT NULL "
        "GROUP BY sprint_id ORDER BY sprint_id"
    ).fetchall()

    mode = "cumulative" if args.cumulative else "per-sprint"
    print(f"{'═' * 80}")
    print(f"  Sprint Evolution — Obsidian-Local TDA ({mode})")
    print(f"{'═' * 80}")
    print(f"  Signal DB: {Path(signal_path).name}")
    print(f"  Sprints: {len(sprints)}  Mode: {mode}\n")

    # Header
    if args.tsv:
        cols = [
            "sprint", "label", "docs", "edges", "simplices",
            "β₀", "β₁", "H0_count", "H0_entropy",
            "H1_count", "maturity", "connected", "stability", "dim_shift",
        ]
        print("\t".join(cols))
    else:
        print(f"  {'Sprint':<7} {'Label':<13} {'Docs':>5} {'Edges':>6} "
              f"{'Simpl':>6} {'β₀':>4} {'β₁':>4} "
              f"{'H0ent':>6} {'H1':>3} "
              f"{'Matur':>6} {'Conn':>6} {'Stab':>6} {'DShft':>5}")
        print(f"  {'─' * 96}")

    results = []
    cumulative_sprints = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for sprint_id, doc_count in sprints:
            if doc_count < 2:
                continue

            if args.cumulative:
                cumulative_sprints.append(sprint_id)
                sprint_list = cumulative_sprints
            else:
                sprint_list = [sprint_id]

            db_path = os.path.join(tmpdir, f"sprint_{sprint_id}.db")
            build_sprint_db(source_conn, sprint_list, db_path)
            metrics = compute_sprint_metrics(db_path)
            os.unlink(db_path)

            if metrics is None:
                continue

            label = sprint_to_label(sprint_id)
            metrics["sprint"] = sprint_id
            metrics["label"] = label
            results.append(metrics)

            if args.tsv:
                vals = [
                    sprint_id, label, str(metrics["nodes"]), str(metrics["edges"]),
                    str(metrics["simplices"]), str(metrics["b0"]), str(metrics["b1"]),
                    str(metrics["h0_count"]), f"{metrics['h0_entropy']:.4f}",
                    str(metrics["h1_count"]),
                    f"{metrics['maturity']:.4f}", f"{metrics['connected']:.4f}",
                    f"{metrics['stability']:.4f}", f"{metrics['dim_shift']:.1f}",
                ]
                print("\t".join(vals))
            else:
                print(
                    f"  {sprint_id:<7} {label:<13} {metrics['nodes']:>5} "
                    f"{metrics['edges']:>6} {metrics['simplices']:>6} "
                    f"{metrics['b0']:>4} {metrics['b1']:>4} "
                    f"{metrics['h0_entropy']:>6.2f} {metrics['h1_count']:>3} "
                    f"{metrics['maturity']:>6.3f} {metrics['connected']:>6.3f} "
                    f"{metrics['stability']:>6.3f} {metrics['dim_shift']:>5.1f}"
                )

    source_conn.close()

    if not args.tsv and len(results) >= 2:
        # Summary: trend analysis
        first = results[0]
        last = results[-1]
        print(f"\n  {'─' * 96}")
        print(f"  TREND ({first['sprint']} → {last['sprint']}):")
        print(f"    Maturity:     {first['maturity']:.3f} → {last['maturity']:.3f}  "
              f"(Δ{last['maturity'] - first['maturity']:+.3f})")
        print(f"    Connectedness: {first['connected']:.3f} → {last['connected']:.3f}  "
              f"(Δ{last['connected'] - first['connected']:+.3f})")
        print(f"    H0 entropy:   {first['h0_entropy']:.3f} → {last['h0_entropy']:.3f}  "
              f"(Δ{last['h0_entropy'] - first['h0_entropy']:+.3f})")

        # Find peak maturity sprint
        peak = max(results, key=lambda r: r["maturity"])
        print(f"    Peak maturity: {peak['sprint']} ({peak['label']}) = {peak['maturity']:.3f}")

        # Find most connected sprint
        most_conn = max(results, key=lambda r: r["connected"])
        print(f"    Most connected: {most_conn['sprint']} ({most_conn['label']}) = {most_conn['connected']:.3f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
