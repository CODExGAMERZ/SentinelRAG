from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore

WIKILINK_CLEAN_RE = re.compile(r"^([^#|]+)")


def clean_wikilink_target(raw_link: str) -> str:
    """Extracts the base note title/path from a Wikilink string."""
    cleaned = raw_link.strip()
    if cleaned.startswith("[[") and cleaned.endswith("]]"):
        cleaned = cleaned[2:-2].strip()
    match = WIKILINK_CLEAN_RE.match(cleaned)
    if not match:
        return cleaned
    return match.group(1).strip()


def resolve_wikilink(
    graph_store: GraphStore,
    raw_link: str,
    source_path: str,
) -> tuple[list[str], bool]:
    """
    Resolves a Wikilink string to a list of target note file paths (keys in GraphStore).
    Returns (resolved_paths, ambiguous_flag).
    """
    target = clean_wikilink_target(raw_link)
    if not target:
        return [], False

    has_folder = "/" in target or "\\" in target
    target_normalized = target.replace("\\", "/")

    target_stem = Path(target_normalized).stem
    candidates = graph_store.get_note_by_title(target_stem)

    if not candidates:
        with graph_store.get_conn() as conn:
            cursor = conn.execute("SELECT path, mtime FROM nodes WHERE path LIKE ?", (f"%{target_normalized}%",))
            candidates = [dict(row) for row in cursor]

    if not candidates:
        return [], False

    if has_folder:
        matched = []
        for cand in candidates:
            cand_path = cand["path"].replace("\\", "/")
            if cand_path.endswith(target_normalized) or cand_path.endswith(target_normalized + ".md"):
                matched.append(cand)
        if matched:
            return [m["path"] for m in matched], len(matched) > 1

    source_dir = Path(source_path).parent.resolve()
    same_dir_candidates = []
    for cand in candidates:
        cand_dir = Path(cand["path"]).parent.resolve()
        if cand_dir == source_dir:
            same_dir_candidates.append(cand)
    
    if len(same_dir_candidates) == 1:
        return [same_dir_candidates[0]["path"]], False
    elif len(same_dir_candidates) > 1:
        candidates = same_dir_candidates

    if len(candidates) > 1:
        candidates_sorted = sorted(candidates, key=lambda x: x["mtime"], reverse=True)
        newest_mtime = candidates_sorted[0]["mtime"]
        ties = [c for c in candidates_sorted if c["mtime"] == newest_mtime]
        if len(ties) == 1:
            return [ties[0]["path"]], False
        return [t["path"] for t in ties], True

    return [candidates[0]["path"]], False
