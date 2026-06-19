from sentinelrag.obsidian.parser import parse_markdown_text


SAMPLE = """# Qwen3

This is a paragraph with #tag and [[Gemma3]].

- first item
- second item
"""


def test_parser_tiers_produce_same_shapes() -> None:
    tier1 = parse_markdown_text(SAMPLE, "note.md", parser_tier="tier1")
    tier2 = parse_markdown_text(SAMPLE, "note.md", parser_tier="tier2")
    tier3 = parse_markdown_text(SAMPLE, "note.md", parser_tier="tier3")
    assert [(b.block_type, b.content) for b in tier1] == [(b.block_type, b.content) for b in tier2]
    assert [(b.block_type, b.content) for b in tier2] == [(b.block_type, b.content) for b in tier3]
