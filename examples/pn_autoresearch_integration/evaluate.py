"""Search quality evaluation: NDCG@k, MRR, P@k.

Self-contained module implementing the same metrics as
kitsap-searchengine-lite/src/evaluate.py, with no external dependencies.
"""

from __future__ import annotations

import json
import math
from typing import Callable


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    DCG = sum(rel_i / log2(i + 2)) for i in [0, k)
    NDCG = DCG / IDCG where IDCG is DCG of the ideal ranking.
    """
    ranked = ranked[:k]

    # DCG: relevance is binary (1 if in relevant set, 0 otherwise)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc in enumerate(ranked) if doc in relevant
    )

    # IDCG: best possible DCG with min(|relevant|, k) relevant docs at top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(ranked: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, doc in enumerate(ranked):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int = 3) -> float:
    """Precision at rank k: fraction of top-k that are relevant."""
    ranked = ranked[:k]
    if not ranked:
        return 0.0
    hits = sum(1 for doc in ranked if doc in relevant)
    return hits / len(ranked)


def evaluate_search_quality(
    search_fn: Callable[[str], list[str]],
    queries: list[dict],
    k: int = 10,
) -> dict:
    """Evaluate search quality over a set of queries.

    Args:
        search_fn: Takes query string, returns list of doc IDs (ranked).
        queries: List of {"query": str, "relevant": list[str]}.
        k: Rank cutoff for NDCG.

    Returns:
        {
            "ndcg_at_10": float,
            "mrr": float,
            "precision_at_3": float,
            "num_queries": int,
        }
    """
    if not queries:
        return {"ndcg_at_10": 0.0, "mrr": 0.0, "precision_at_3": 0.0, "num_queries": 0}

    ndcg_scores = []
    mrr_scores = []
    p3_scores = []

    for q in queries:
        query_text = q["query"]
        relevant_set = set(q["relevant"])
        ranked = search_fn(query_text)

        ndcg_scores.append(ndcg_at_k(ranked, relevant_set, k=k))
        mrr_scores.append(mrr(ranked, relevant_set))
        p3_scores.append(precision_at_k(ranked, relevant_set, k=3))

    return {
        "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "precision_at_3": sum(p3_scores) / len(p3_scores),
        "num_queries": len(queries),
    }


def load_queries(path: str) -> list[dict]:
    """Load queries from JSONL file."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries
