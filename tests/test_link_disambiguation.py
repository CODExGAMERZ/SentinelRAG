import time
from pathlib import Path
from sentinelrag.graph.graph_store import GraphStore
from sentinelrag.graph.link_resolver import resolve_wikilink


def test_link_disambiguation_rules(tmp_path: Path) -> None:
    graph_store = GraphStore(tmp_path, "test.db")
    
    # Notes:
    # 1. Research/Qwen.md
    # 2. Archive/Qwen.md
    # 3. Archive/Other.md
    
    graph_store.upsert_note("Research/Qwen.md", "Qwen", time.time() - 100)
    graph_store.upsert_note("Archive/Qwen.md", "Qwen", time.time()) # newer
    graph_store.upsert_note("Archive/Other.md", "Other", time.time())
    
    # Rule 1: Exact path match if folder included
    res, amb = resolve_wikilink(graph_store, "[[Research/Qwen]]", "Archive/Other.md")
    assert res == ["Research/Qwen.md"]
    assert not amb
    
    # Rule 2: Prefer same folder
    res, amb = resolve_wikilink(graph_store, "[[Qwen]]", "Archive/Other.md")
    assert res == ["Archive/Qwen.md"]
    assert not amb

    # Rule 3: Prefer newest mtime (if no folder context and not same folder)
    # Let's test from Root/Other.md
    res, amb = resolve_wikilink(graph_store, "[[Qwen]]", "Root/Other.md")
    assert res == ["Archive/Qwen.md"]
    assert not amb
