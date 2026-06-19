from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore


def traverse_graph(
    graph_store: GraphStore,
    seed_paths: list[str],
    max_depth: int = 1,
) -> dict[str, int]:
    """
    Performs a Breadth-First Search (BFS) starting from seed_paths up to max_depth.
    Returns a dict mapping traversed node paths to their shortest distance (depth) from the seeds.
    """
    if not seed_paths or max_depth < 0:
        return {}

    adj: dict[str, list[str]] = {}
    edges = graph_store.get_all_edges()
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)  # treat graph as undirected for Wikilinks traversal

    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for path in seed_paths:
        queue.append((path, 0))
        visited[path] = 0

    while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = adj.get(curr, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return visited
