from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..types import MarkdownBlockIR

TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9/_-]+)")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_RE = re.compile(r"^\s*[-*+]\s+")


@dataclass(frozen=True, slots=True)
class ParserProbeResult:
    tier: str
    engine: str


def probe_parser_tier() -> ParserProbeResult:
    """Probes the system to find the highest functional parser tier."""
    try:
        import tree_sitter  # type: ignore  # noqa: F401
        return ParserProbeResult(tier="tier1", engine="tree-sitter")
    except Exception:
        pass

    try:
        import markdown_it  # type: ignore  # noqa: F401
        return ParserProbeResult(tier="tier2", engine="markdown-it-py")
    except Exception:
        pass

    return ParserProbeResult(tier="tier3", engine="regex")


def parse_markdown_file(path: Path, parser_tier: str | None = None) -> list[MarkdownBlockIR]:
    return parse_markdown_text(path.read_text(encoding="utf-8", errors="replace"), str(path), parser_tier=parser_tier)


def parse_markdown_text(text: str, source_path: str, parser_tier: str | None = None) -> list[MarkdownBlockIR]:
    tier = (parser_tier or probe_parser_tier().tier).lower()
    parser = _select_parser(tier)
    return parser(text, source_path)


def _select_parser(tier: str) -> Callable[[str, str], list[MarkdownBlockIR]]:
    if tier in {"tier1", "tree-sitter"}:
        try:
            return _tree_sitter_parse
        except Exception:
            pass
    if tier in {"tier2", "markdown-it-py"}:
        try:
            return _markdown_it_parse
        except Exception:
            pass
    return _regex_structural_parse


def _tree_sitter_parse(text: str, source_path: str) -> list[MarkdownBlockIR]:
    try:
        raise NotImplementedError("tree-sitter-markdown grammar not compiled; falling back.")
    except Exception:
        return _markdown_it_parse(text, source_path)


def _markdown_it_parse(text: str, source_path: str) -> list[MarkdownBlockIR]:
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    tokens = md.parse(text)
    blocks: list[MarkdownBlockIR] = []
    headers: list[str] = []
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token.type == "heading_open":
            level = int(token.tag[1])  # 'h1' -> 1
            i += 1
            inline_token = tokens[i]
            title = inline_token.content.strip()
            
            headers[:] = headers[: level - 1]
            headers.append(title)
            blocks.append(_build_block(source_path, "heading", title, headers))
            
        elif token.type == "paragraph_open":
            i += 1
            inline_token = tokens[i]
            content = inline_token.content.strip()
            if content:
                blocks.append(_build_block(source_path, "paragraph", content, headers))
                
        elif token.type == "fence":
            content = token.content.strip()
            if content:
                blocks.append(_build_block(source_path, "code_block", content, headers))
                
        elif token.type == "list_item_open":
            item_content = []
            depth = 1
            while i + 1 < len(tokens) and depth > 0:
                i += 1
                t = tokens[i]
                if t.type == "list_item_open":
                    depth += 1
                elif t.type == "list_item_close":
                    depth -= 1
                elif t.type == "inline" and t.content.strip():
                    item_content.append(t.content.strip())
            
            content = "\n".join(item_content).strip()
            if content:
                blocks.append(_build_block(source_path, "list_item", content, headers))
                
        i += 1

    if not blocks and text.strip():
        return _regex_structural_parse(text, source_path)
        
    return blocks


def _regex_structural_parse(text: str, source_path: str) -> list[MarkdownBlockIR]:
    lines = text.splitlines()
    blocks: list[MarkdownBlockIR] = []
    headers: list[str] = []
    code_fence = False
    buffer: list[str] = []
    current_type = "paragraph"

    def flush() -> None:
        nonlocal buffer, current_type
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            current_type = "paragraph"
            return
        blocks.append(_build_block(source_path, current_type, content, headers))
        buffer = []
        current_type = "paragraph"

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if line.strip().startswith("```"):
            if code_fence:
                buffer.append(line)
                current_type = "code_block"
                flush()
                code_fence = False
                continue
            flush()
            code_fence = True
            current_type = "code_block"
            buffer.append(line)
            continue
        if code_fence:
            buffer.append(line)
            continue

        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headers[:] = headers[: level - 1]
            headers.append(title)
            blocks.append(_build_block(source_path, "heading", title, headers))
            continue

        if not line.strip():
            flush()
            continue

        if LIST_RE.match(line):
            flush()
            current_type = "list_item"
            clean_line = LIST_RE.sub("", line).strip()
            buffer.append(clean_line)
            continue

        if current_type != "paragraph":
            flush()
        current_type = "paragraph"
        buffer.append(line)

    flush()
    return blocks


def _build_block(source_path: str, block_type: str, content: str, headers: list[str]) -> MarkdownBlockIR:
    content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    header_path = " > ".join(headers[:-1] if block_type == "heading" else headers)
    fingerprint = content[:64]
    
    block_seed = f"{header_path}|{block_type}|{fingerprint}"
    block_id = hashlib.sha256(block_seed.encode("utf-8", errors="replace")).hexdigest()[:24]
    
    tags = [f"#{match.group(1)}" for match in TAG_RE.finditer(content)]
    links = [match.group(1).strip() for match in WIKILINK_RE.finditer(content)]
    
    return MarkdownBlockIR(
        block_id=block_id,
        content_hash=content_hash,
        source_path=source_path,
        block_type=block_type,
        content=content,
        headers=list(headers),
        tags=tags,
        links=links,
    )
