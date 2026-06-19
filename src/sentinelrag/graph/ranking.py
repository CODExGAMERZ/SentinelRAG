from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore


def compute_pagerank(
    graph_store: GraphStore,
    alpha: float = 0.85,
    max_iter: int = 20,
) -> dict[str, float]:
    """
    Computes PageRank centrality scores for all note nodes in the GraphStore.
    """
    notes = graph_store.get_all_notes()
    nodes = [note["path"] for note in notes]
    if not nodes:
        return {}

    edges = graph_store.get_all_edges()
    n_nodes = len(nodes)

    out_degree: dict[str, int] = {node: 0 for node in nodes}
    in_edges: dict[str, list[str]] = {node: [] for node in nodes}

    for src, dst in edges:
        if src in out_degree and dst in in_edges:
            out_degree[src] += 1
            in_edges[dst].append(src)
        if dst in out_degree and src in in_edges:
            out_degree[dst] += 1
            in_edges[src].append(dst)

    pr = {node: 1.0 / n_nodes for node in nodes}

    for _ in range(max_iter):
        new_pr = {}
        dangling_sum = sum(pr[node] for node in nodes if out_degree[node] == 0)
        dangling_share = dangling_sum / n_nodes
        
        for node in nodes:
            rank = (1.0 - alpha) / n_nodes + alpha * dangling_share
            for parent in in_edges[node]:
                rank += alpha * pr[parent] / out_degree[parent]
            new_pr[node] = rank
        pr = new_pr

    return pr


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """
    Normalizes a dictionary of scores into [0.0, 1.0] using Min-Max scaling.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    max_val = max(vals)
    min_val = min(vals)
    if max_val == min_val:
        return {k: 1.0 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def compute_recency_score(
    mtime: float,
    is_evergreen: bool,
    decay_days: float = 90.0,
) -> float:
    """
    Computes the temporal relevance / recency score.
    Evergreen notes have a fixed weight of 1.0 (no decay).
    Fleeting notes decay exponentially: exp(-days_old / decay_days).
    """
    if is_evergreen:
        return 1.0
    now = time.time()
    age_seconds = max(0.0, now - mtime)
    age_days = age_seconds / (24.0 * 3600.0)
    return math.exp(-age_days / decay_days)
