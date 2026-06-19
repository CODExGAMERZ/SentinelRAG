import time
from pathlib import Path
from sentinelrag.graph.graph_store import GraphStore
from sentinelrag.graph.link_resolver import resolve_wikilink


def test_link_disambiguation_rules(tmp_path: Path) -> None:
    graph_store = GraphStore(tmp_path, "test.db")
    
    
    graph_store.upsert_note("Research/Qwen.md", "Qwen", time.time() - 100)
    graph_store.upsert_note("Archive/Qwen.md", "Qwen", time.time()) # newer
    graph_store.upsert_note("Archive/Other.md", "Other", time.time())
    
    res, amb = resolve_wikilink(graph_store, "[[Research/Qwen]]", "Archive/Other.md")
    assert res == ["Research/Qwen.md"]
    assert not amb
    
    res, amb = resolve_wikilink(graph_store, "[[Qwen]]", "Archive/Other.md")
    assert res == ["Archive/Qwen.md"]
    assert not amb

    res, amb = resolve_wikilink(graph_store, "[[Qwen]]", "Root/Other.md")
    assert res == ["Archive/Qwen.md"]
    assert not amb
