"""TDA feature cache: save/load precomputed features to SQLite.

Stores the expensive pipeline output (features + doc_ids + config hash)
so subsequent runs skip VR construction and PH computation.

Schema matches the design doc tda_features table with an added
metadata table for cache invalidation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tda_cache_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tda_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    dimension INTEGER,
    feature_name TEXT NOT NULL,
    feature_value REAL NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tda_features_doc ON tda_features(doc_id);
CREATE INDEX IF NOT EXISTS idx_tda_features_type ON tda_features(feature_type);
"""


def config_hash(
    signal_db_path: str,
    epsilon_max: float,
    max_dimension: int,
    num_scales: int,
    max_nodes: int,
    use_graph_filtration: bool = True,
) -> str:
    """Deterministic hash of pipeline config for cache invalidation."""
    key = json.dumps({
        "signal_db": str(Path(signal_db_path).resolve()),
        "signal_db_size": Path(signal_db_path).stat().st_size,
        "epsilon_max": epsilon_max,
        "max_dimension": max_dimension,
        "num_scales": num_scales,
        "max_nodes": max_nodes,
        "graph_filtration": use_graph_filtration,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_cache(cache_path: str, expected_hash: str) -> dict | None:
    """Load cached features if they exist and config matches.

    Returns the same dict shape as run_tda_pipeline(), or None on miss.
    """
    if not Path(cache_path).is_file():
        return None

    conn = sqlite3.connect(cache_path)
    try:
        stored_hash = conn.execute(
            "SELECT value FROM tda_cache_meta WHERE key='config_hash'"
        ).fetchone()
        if not stored_hash or stored_hash[0] != expected_hash:
            return None

        # Load doc_ids
        doc_ids_json = conn.execute(
            "SELECT value FROM tda_cache_meta WHERE key='doc_ids'"
        ).fetchone()
        if not doc_ids_json:
            return None
        doc_ids = json.loads(doc_ids_json[0])

        # Load scalar features
        rows = conn.execute(
            "SELECT feature_type, feature_name, feature_value FROM tda_features "
            "WHERE doc_id = '_corpus_'"
        ).fetchall()

        scalars = {}
        for ftype, fname, fval in rows:
            scalars[f"{ftype}.{fname}"] = fval

        # Load full results JSON
        full_json = conn.execute(
            "SELECT value FROM tda_cache_meta WHERE key='full_results'"
        ).fetchone()
        if not full_json:
            return None

        result = json.loads(full_json[0])
        result["doc_ids"] = doc_ids
        return result

    except (sqlite3.OperationalError, json.JSONDecodeError):
        return None
    finally:
        conn.close()


def save_cache(cache_path: str, cfg_hash: str, features: dict) -> None:
    """Save pipeline features to cache file."""
    conn = sqlite3.connect(cache_path)
    conn.executescript(_SCHEMA)

    # Clear old data
    conn.execute("DELETE FROM tda_cache_meta")
    conn.execute("DELETE FROM tda_features")

    # Save config hash
    conn.execute(
        "INSERT INTO tda_cache_meta VALUES (?, ?)", ("config_hash", cfg_hash)
    )

    # Save doc_ids
    conn.execute(
        "INSERT INTO tda_cache_meta VALUES (?, ?)",
        ("doc_ids", json.dumps(features["doc_ids"])),
    )

    # Save corpus-level scalar features
    corpus_scalars = [
        ("maturity", "maturity_score", features["maturity_score"]),
        ("persistence", "norm_entropy", features["norm_entropy"]),
        ("betti", "stability", features["betti_stability"]),
    ]
    for ftype, fname, fval in corpus_scalars:
        conn.execute(
            "INSERT INTO tda_features (doc_id, feature_type, dimension, feature_name, feature_value) "
            "VALUES (?, ?, NULL, ?, ?)",
            ("_corpus_", ftype, fname, fval),
        )

    # Save full results as JSON (for betti_result, persist_result, maturity_result)
    serializable = {
        k: v for k, v in features.items()
        if k not in ("doc_ids",)  # doc_ids saved separately
    }
    conn.execute(
        "INSERT INTO tda_cache_meta VALUES (?, ?)",
        ("full_results", json.dumps(serializable)),
    )

    conn.commit()
    conn.close()
