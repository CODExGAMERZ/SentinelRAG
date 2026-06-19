import time
from pathlib import Path
from sentinelrag.graph.graph_store import GraphStore
from sentinelrag.graph.ranking import compute_pagerank, min_max_normalize, compute_recency_score


def test_pagerank_computation(tmp_path: Path) -> None:
    graph_store = GraphStore(tmp_path, "test.db")
    
    graph_store.upsert_note("A.md", "A", time.time())
    graph_store.upsert_note("B.md", "B", time.time())
    graph_store.upsert_note("C.md", "C", time.time())
    
    graph_store.add_edge("A.md", "B.md")
    graph_store.add_edge("B.md", "C.md")
    
    pr = compute_pagerank(graph_store)
    assert len(pr) == 3
    assert pr["B.md"] > 0


def test_min_max_normalization() -> None:
    scores = {"A": 10.0, "B": 20.0, "C": 30.0}
    normalized = min_max_normalize(scores)
    assert normalized["A"] == 0.0
    assert normalized["B"] == 0.5
    assert normalized["C"] == 1.0


def test_recency_score_decay() -> None:
    now = time.time()
    assert compute_recency_score(now - 1000000, is_evergreen=True) == 1.0
    
    fresh = compute_recency_score(now, is_evergreen=False)
    assert abs(fresh - 1.0) < 1e-4
    
    old = compute_recency_score(now - (90 * 24 * 3600), is_evergreen=False)
    assert round(old, 4) == 0.3679
