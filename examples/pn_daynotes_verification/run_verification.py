#!/usr/bin/env python3
"""PN Daynotes NDCG Verification: real corpus baseline vs TDA experiment.

Runs the full TDA pipeline on a real SignalDB and measures NDCG@10 impact
against the PN Daynotes corpus with pinned queries.

Usage:
    cd pn-tda
    python3 examples/pn_daynotes_verification/run_verification.py
    python3 examples/pn_daynotes_verification/run_verification.py --tda-weight 0.2
    python3 examples/pn_daynotes_verification/run_verification.py --signal-db ~/azure-wiki-analysis/output/obsidian-signals.db
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

# Ensure imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.pn_autoresearch_integration.evaluate import (
    evaluate_search_quality,
    load_queries,
)
from examples.pn_daynotes_verification.cache import (
    config_hash,
    load_cache,
    save_cache,
)
from pn_tda.adapters.signal_db import SignalDBGraph
from pn_tda.core.filtration import VietorisRipsBuilder
from pn_tda.core.persistence import PersistentHomology
from pn_tda.features.betti import BettiNumberExtractor
from pn_tda.features.maturity import ThreadMaturityScorer
from pn_tda.features.persistence import PersistenceFeatureExtractor

WORD_RE = re.compile(r"\b[a-z]{3,}\b")

# Default paths (auto-detect)
DEFAULT_CORPUS_PATHS = [
    os.path.expanduser(
        "~/pn-monorepo/pn-monorepo/projects/kitsap-searchengine-lite/data/pn-daynotes.db"
    ),
]
DEFAULT_SIGNAL_PATHS = [
    os.path.expanduser("~/azure-wiki-analysis/output/signals.db"),
    os.path.expanduser("~/azure-wiki-analysis/output/obsidian-signals.db"),
]
DEFAULT_QUERIES_PATHS = [
    str(
        Path(__file__).resolve().parents[2].parent
        / "pn-autoresearch/queries/pn-daynotes.jsonl"
    ),
    os.path.expanduser(
        "~/pn-monorepo/pn-monorepo/projects/pn-autoresearch/queries/pn-daynotes.jsonl"
    ),
    # Fall back to the repo copy
    str(
        Path(__file__).resolve().parents[2].parent.parent
        / "pn-monorepo/projects/pn-autoresearch/queries/pn-daynotes.jsonl"
    ),
]


def find_path(candidates: list[str], env_var: str, label: str) -> str:
    """Find first existing path from candidates or env var."""
    env_val = os.environ.get(env_var, "")
    if env_val and os.path.isfile(env_val):
        return env_val
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find {label}. Tried: {env_var} env var, {candidates}. "
        f"Set {env_var} or use --{label.lower().replace(' ', '-')} flag."
    )


def keyword_jaccard(query: str, text: str) -> float:
    q_tokens = set(WORD_RE.findall(query.lower()))
    t_tokens = set(WORD_RE.findall(text.lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens | t_tokens)


def normalize_path(source_path: str) -> str:
    """Strip the wiki root prefix to match query relevance paths."""
    prefixes = [
        "/mnt/c/projects/Daynotes.wiki/",
        "/mnt/c/Projects/Daynotes.wiki/",
        "C:/projects/Daynotes.wiki/",
    ]
    for prefix in prefixes:
        if source_path.startswith(prefix):
            return source_path[len(prefix) :]
    return source_path


def subsample_signal_db(signal_db_path: str, max_nodes: int, tmp_dir: str) -> str:
    """Create a subsampled signal DB with the most signal-rich documents.

    Selects docs with the highest signal+edge count to preserve the most
    topologically interesting subgraph.
    """
    import shutil

    conn = sqlite3.connect(signal_db_path)
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    if total <= max_nodes:
        conn.close()
        return signal_db_path  # No subsampling needed

    # Rank docs by signal+edge density
    top_docs = conn.execute("""
        SELECT d.doc_id,
               COALESCE(s.sig_count, 0) + COALESCE(e.edge_count, 0) AS density
        FROM documents d
        LEFT JOIN (SELECT doc_id, COUNT(*) AS sig_count FROM signals GROUP BY doc_id) s
            ON d.doc_id = s.doc_id
        LEFT JOIN (SELECT source_doc_id AS doc_id, COUNT(*) AS edge_count FROM edges GROUP BY source_doc_id) e
            ON d.doc_id = e.doc_id
        ORDER BY density DESC
        LIMIT ?
    """, (max_nodes,)).fetchall()
    keep_ids = {row[0] for row in top_docs}
    conn.close()

    # Build subsampled DB
    sub_path = os.path.join(tmp_dir, "signals_subsample.db")
    src = sqlite3.connect(signal_db_path)
    dst = sqlite3.connect(sub_path)

    # Copy schema (skip internal SQLite tables)
    for row in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%'"
    ):
        dst.execute(row[0])

    # Copy rows for selected docs
    placeholders = ",".join("?" for _ in keep_ids)
    id_list = list(keep_ids)

    for row in src.execute(f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", id_list):
        dst.execute(f"INSERT INTO documents VALUES ({','.join('?' for _ in row)})", tuple(row))

    for row in src.execute(f"SELECT * FROM signals WHERE doc_id IN ({placeholders})", id_list):
        dst.execute(f"INSERT INTO signals VALUES ({','.join('?' for _ in row)})", tuple(row))

    for row in src.execute(f"SELECT * FROM edges WHERE source_doc_id IN ({placeholders})", id_list):
        dst.execute(f"INSERT INTO edges VALUES ({','.join('?' for _ in row)})", tuple(row))

    dst.commit()
    src.close()
    dst.close()

    print(f"   Subsampled: {total} → {len(keep_ids)} docs (top by signal density)")
    return sub_path


def run_tda_pipeline(
    signal_db_path: str,
    epsilon_max: float,
    max_dimension: int,
    num_scales: int,
    max_nodes: int = 500,
    tmp_dir: str | None = None,
) -> dict:
    """Run TDA pipeline on signal DB and return per-doc features + corpus features."""
    import math
    import tempfile

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()

    print(f"   Loading SignalDB: {signal_db_path}")

    # Subsample if needed
    effective_path = subsample_signal_db(signal_db_path, max_nodes, tmp_dir)

    graph = SignalDBGraph(effective_path)
    doc_ids = list(graph.nodes())
    print(f"   Graph: {len(doc_ids)} nodes")

    print(f"   Building VR complex (ε={epsilon_max}, dim≤{max_dimension})...")
    t0 = time.time()
    builder = VietorisRipsBuilder(
        epsilon_max=epsilon_max, max_dimension=max_dimension
    )
    st = builder.build(graph)
    print(f"   VR complex: {st.num_simplices()} simplices ({time.time()-t0:.1f}s)")

    print("   Computing persistent homology...")
    t0 = time.time()
    intervals = PersistentHomology().compute(st)
    print(f"   Persistence intervals: {len(intervals)} ({time.time()-t0:.1f}s)")

    # Extract features
    betti_ext = BettiNumberExtractor()
    persist_ext = PersistenceFeatureExtractor()
    maturity_scorer = ThreadMaturityScorer()

    betti_result = betti_ext.extract(
        intervals, num_scales=num_scales, epsilon_max=epsilon_max,
        max_dimension=max_dimension,
    )
    persist_result = persist_ext.extract(intervals, max_dimension=max_dimension)
    maturity_result = maturity_scorer.score(
        intervals, graph, num_scales=num_scales, epsilon_max=epsilon_max,
    )

    # Compute composite
    maturity_score = maturity_result.get("maturity_score", 0.0)
    dim0_entropy = persist_result.get("dim_0_entropy", 0.0)
    dim0_count = persist_result.get("dim_0_count", 1)
    max_entropy = math.log(max(dim0_count, 2))
    norm_entropy = min(dim0_entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
    betti_std = betti_result["summary"].get("betti_0_std", 0.0)
    betti_mean = betti_result["summary"].get("betti_0_mean", 1.0)
    betti_stability = max(
        0.0, 1.0 - (betti_std / betti_mean if betti_mean > 0 else 0.0)
    )

    print(f"   Features: maturity={maturity_score:.4f}, "
          f"entropy={norm_entropy:.4f}, stability={betti_stability:.4f}")

    return {
        "doc_ids": doc_ids,
        "maturity_score": maturity_score,
        "norm_entropy": norm_entropy,
        "betti_stability": betti_stability,
        "betti_result": betti_result,
        "persist_result": persist_result,
        "maturity_result": maturity_result,
    }


def make_baseline_search(corpus_conn: sqlite3.Connection):
    """Keyword Jaccard search — returns normalized doc paths."""
    rows = corpus_conn.execute(
        "SELECT c.chunk_id, c.content, c.heading, d.source_path "
        "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id"
    ).fetchall()

    def search(query: str) -> list[str]:
        scored: list[tuple[str, float]] = []
        seen_paths: set[str] = set()
        for r in rows:
            text = f"{r['heading'] or ''} {r['content']}"
            score = keyword_jaccard(query, text)
            path = normalize_path(r["source_path"])
            if score > 0 and path not in seen_paths:
                scored.append((path, score))
                seen_paths.add(path)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scored[:10]]

    return search


def make_tda_search(
    corpus_conn: sqlite3.Connection,
    tda_features: dict,
    signal_doc_ids: set[str],
    tda_weight: float,
    maturity_weight: float = 0.5,
    persistence_weight: float = 0.3,
    betti_weight: float = 0.2,
):
    """Keyword Jaccard + TDA composite fusion — returns normalized doc paths."""
    rows = corpus_conn.execute(
        "SELECT c.chunk_id, c.doc_id, c.content, c.heading, d.source_path "
        "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id"
    ).fetchall()

    # Compute composite score
    total_w = maturity_weight + persistence_weight + betti_weight
    if total_w == 0:
        total_w = 1.0
    composite = (
        maturity_weight * tda_features["maturity_score"]
        + persistence_weight * tda_features["norm_entropy"]
        + betti_weight * tda_features["betti_stability"]
    ) / total_w

    keyword_weight = 1.0 - tda_weight

    def search(query: str) -> list[str]:
        scored: list[tuple[str, float]] = []
        seen_paths: set[str] = set()
        for r in rows:
            text = f"{r['heading'] or ''} {r['content']}"
            kw_score = keyword_jaccard(query, text)
            # Doc gets TDA boost if it's in the signal DB
            tda_score = composite if r["doc_id"] in signal_doc_ids else 0.0
            combined = keyword_weight * kw_score + tda_weight * tda_score
            path = normalize_path(r["source_path"])
            if combined > 0 and path not in seen_paths:
                scored.append((path, combined))
                seen_paths.add(path)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scored[:10]]

    return search


def main():
    parser = argparse.ArgumentParser(
        description="PN Daynotes NDCG Verification: baseline vs TDA"
    )
    parser.add_argument("--corpus-db", help="Path to corpus SQLite (chunks + documents)")
    parser.add_argument("--signal-db", help="Path to azure-wiki-analysis signal SQLite")
    parser.add_argument("--queries", help="Path to queries JSONL")
    parser.add_argument("--tda-weight", type=float, default=float(os.environ.get("TDA_WEIGHT", "0.3")),
                        help="TDA weight in fusion (default: 0.3)")
    parser.add_argument("--epsilon-max", type=float, default=float(os.environ.get("TDA_EPSILON_MAX", "1.0")),
                        help="VR complex epsilon (default: 1.0)")
    parser.add_argument("--max-dimension", type=int, default=int(os.environ.get("TDA_MAX_DIM", "2")),
                        help="Max simplex dimension (default: 2)")
    parser.add_argument("--num-scales", type=int, default=int(os.environ.get("TDA_NUM_SCALES", "10")),
                        help="Filtration scale count (default: 10)")
    parser.add_argument("--max-nodes", type=int, default=int(os.environ.get("TDA_MAX_NODES", "300")),
                        help="Max signal DB nodes for TDA (subsamples by density, default: 300)")
    parser.add_argument("--cache", default=os.environ.get("TDA_CACHE", ""),
                        help="Path to cache file for TDA features (skips pipeline on hit)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute even if cache exists")
    args = parser.parse_args()

    # Resolve paths
    corpus_path = args.corpus_db or find_path(DEFAULT_CORPUS_PATHS, "CORPUS_DB", "corpus DB")
    signal_path = args.signal_db or find_path(DEFAULT_SIGNAL_PATHS, "SIGNAL_DB", "signal DB")
    queries_path = args.queries or find_path(DEFAULT_QUERIES_PATHS, "QUERIES_PATH", "queries JSONL")

    # Load queries
    queries = load_queries(queries_path)

    # Corpus stats
    corpus_conn = sqlite3.connect(corpus_path)
    corpus_conn.row_factory = sqlite3.Row
    doc_count = corpus_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = corpus_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # Signal stats
    sig_conn = sqlite3.connect(signal_path)
    sig_doc_count = sig_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    sig_signal_count = sig_conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    sig_conn.close()

    print("=" * 60)
    print("PN Daynotes NDCG Verification")
    print("=" * 60)
    print(f"\nCorpus:  {Path(corpus_path).name} ({doc_count} docs, {chunk_count} chunks)")
    print(f"Signals: {Path(signal_path).name} ({sig_doc_count} docs, {sig_signal_count} signals)")
    print(f"Queries: {Path(queries_path).name} ({len(queries)} queries)")
    print(f"Config:  tda_weight={args.tda_weight}, ε={args.epsilon_max}, "
          f"dim≤{args.max_dimension}, scales={args.num_scales}, "
          f"max_nodes={args.max_nodes}")

    # --- Baseline ---
    print("\n1. Running BASELINE (keyword Jaccard only)...")
    t0 = time.time()
    baseline_search = make_baseline_search(corpus_conn)
    baseline_metrics = evaluate_search_quality(baseline_search, queries)
    baseline_time = time.time() - t0
    print(f"   NDCG@10={baseline_metrics['ndcg_at_10']:.4f} "
          f"MRR={baseline_metrics['mrr']:.4f} "
          f"P@3={baseline_metrics['precision_at_3']:.4f} "
          f"({baseline_time:.1f}s)")

    # --- TDA Pipeline (with cache) ---
    cache_path = args.cache or os.path.join(
        os.path.dirname(signal_path), "tda_cache.db"
    )
    cfg_hash = config_hash(
        signal_path, args.epsilon_max, args.max_dimension,
        args.num_scales, args.max_nodes,
    )

    tda_features = None
    if not args.no_cache:
        tda_features = load_cache(cache_path, cfg_hash)
        if tda_features:
            print(f"\n2. TDA features loaded from cache ({cache_path})")

    if tda_features is None:
        print("\n2. Running TDA pipeline...")
        t0 = time.time()
        tda_features = run_tda_pipeline(
            signal_path, args.epsilon_max, args.max_dimension, args.num_scales,
            max_nodes=args.max_nodes,
        )
        tda_pipeline_time = time.time() - t0
        print(f"   Pipeline complete ({tda_pipeline_time:.1f}s)")

        # Save to cache
        save_cache(cache_path, cfg_hash, tda_features)
        print(f"   Cached to {cache_path}")
    else:
        tda_pipeline_time = 0.0

    signal_doc_ids = set(tda_features["doc_ids"])

    # --- TDA Experiment ---
    kw_pct = int((1.0 - args.tda_weight) * 100)
    tda_pct = int(args.tda_weight * 100)
    print(f"\n3. Running TDA experiment (keyword {kw_pct}% + TDA {tda_pct}%)...")
    t0 = time.time()
    tda_search = make_tda_search(
        corpus_conn, tda_features, signal_doc_ids, args.tda_weight,
    )
    tda_metrics = evaluate_search_quality(tda_search, queries)
    tda_time = time.time() - t0
    print(f"   NDCG@10={tda_metrics['ndcg_at_10']:.4f} "
          f"MRR={tda_metrics['mrr']:.4f} "
          f"P@3={tda_metrics['precision_at_3']:.4f} "
          f"({tda_time:.1f}s)")

    corpus_conn.close()

    # --- Results ---
    delta_ndcg = tda_metrics["ndcg_at_10"] - baseline_metrics["ndcg_at_10"]
    delta_mrr = tda_metrics["mrr"] - baseline_metrics["mrr"]
    delta_p3 = tda_metrics["precision_at_3"] - baseline_metrics["precision_at_3"]

    if delta_ndcg > 0.01:
        decision = "KEEP"
    elif delta_ndcg < -0.02:
        decision = "DISCARD"
    else:
        decision = "NEUTRAL"

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Baseline':>10} {'TDA':>10} {'Delta':>10}")
    print("-" * 52)
    print(f"{'NDCG@10':<20} {baseline_metrics['ndcg_at_10']:>10.4f} "
          f"{tda_metrics['ndcg_at_10']:>10.4f} {delta_ndcg:>+10.4f}")
    print(f"{'MRR':<20} {baseline_metrics['mrr']:>10.4f} "
          f"{tda_metrics['mrr']:>10.4f} {delta_mrr:>+10.4f}")
    print(f"{'P@3':<20} {baseline_metrics['precision_at_3']:>10.4f} "
          f"{tda_metrics['precision_at_3']:>10.4f} {delta_p3:>+10.4f}")
    print(f"{'Queries':<20} {len(queries):>10d}")
    print(f"{'TDA pipeline (s)':<20} {tda_pipeline_time:>10.1f}")
    print(f"{'Eval time (s)':<20} {baseline_time:>10.1f} {tda_time:>10.1f}")
    print(f"\nΔNDCG@10 = {delta_ndcg:+.4f}  →  {decision}")

    # TSV output
    print("\n--- TSV (for results.tsv) ---")
    print("run_tag\tconfig\tsignal_db\tndcg_at_10\tmrr\tprecision_at_3\t"
          "tda_pipeline_s\teval_s\tstatus\tdescription")
    print(f"baseline\tkeyword_only\t-\t{baseline_metrics['ndcg_at_10']:.4f}\t"
          f"{baseline_metrics['mrr']:.4f}\t{baseline_metrics['precision_at_3']:.4f}\t"
          f"-\t{baseline_time:.2f}\tBASELINE\tKeyword Jaccard only")
    print(f"tda-pn\tkeyword+tda\t{Path(signal_path).name}\t"
          f"{tda_metrics['ndcg_at_10']:.4f}\t{tda_metrics['mrr']:.4f}\t"
          f"{tda_metrics['precision_at_3']:.4f}\t{tda_pipeline_time:.2f}\t"
          f"{tda_time:.2f}\t{decision}\t"
          f"Keyword {kw_pct}% + TDA {tda_pct}% (e={args.epsilon_max})")

    return 0 if decision != "DISCARD" else 1


if __name__ == "__main__":
    sys.exit(main())
