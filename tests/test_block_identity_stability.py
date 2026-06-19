from sentinelrag.obsidian.parser import parse_markdown_text


def test_block_identity_stability_insert_delete() -> None:
    content1 = (
        "# Heading 1\n\n"
        "This is paragraph 1.\n\n"
        "# Heading 2\n\n"
        "This is paragraph 2.\n"
    )
    blocks1 = parse_markdown_text(content1, "test.md", "regex")
    assert len(blocks1) == 4

    # Insert a block at the start
    content2 = (
        "New paragraph inserted.\n\n"
        "# Heading 1\n\n"
        "This is paragraph 1.\n\n"
        "# Heading 2\n\n"
        "This is paragraph 2.\n"
    )
    blocks2 = parse_markdown_text(content2, "test.md", "regex")
    assert len(blocks2) == 5

    # Check that block_ids of unchanged paragraphs remain stable
    # Blocks 1 & 2 of blocks2 should match Blocks 0 & 1 of blocks1
    b1_p1 = [b for b in blocks1 if b.content == "This is paragraph 1."][0]
    b2_p1 = [b for b in blocks2 if b.content == "This is paragraph 1."][0]
    assert b1_p1.block_id == b2_p1.block_id

    b1_h2 = [b for b in blocks1 if b.content == "Heading 2"][0]
    b2_h2 = [b for b in blocks2 if b.content == "Heading 2"][0]
    assert b1_h2.block_id == b2_h2.block_id
