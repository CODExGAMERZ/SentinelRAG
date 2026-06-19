from sentinelrag.agents.evidence_merger import merge_evidence
from sentinelrag.types import Evidence
from sentinelrag.graph.graph_store import GraphStore


def test_merges_overlapping_hits_as_hybrid(tmp_path) -> None:
    graph_store = GraphStore(tmp_path, "test.db")
    graph_store.upsert_note("a.md", "a", 123.45)
    
    vector_hits = [
        Evidence(chunk_id="c1", doc_id="d1", source_path="a.md", text="alpha", score=0.8),
    ]
    graph_hits = [
        Evidence(chunk_id="c1", doc_id="d1", source_path="a.md", text="alpha", score=0.6, facts=["Entity"]),
    ]
    merged = merge_evidence(vector_hits, graph_hits, graph_store, query_is_temporal=False)
    assert len(merged) == 1
    assert merged[0].source_type == "hybrid"
