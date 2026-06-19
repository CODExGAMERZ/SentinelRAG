from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore

# Wikilink pattern: [[TargetNoteName#HeaderSection|AliasText]]
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

    # 1. Exact path match (or suffix match if folder is specified)
    # Check if target contains a slash, indicating a path prefix/suffix
    has_folder = "/" in target or "\\" in target
    target_normalized = target.replace("\\", "/")

    # Fetch notes matching the filename stem (basename without extension)
    target_stem = Path(target_normalized).stem
    candidates = graph_store.get_note_by_title(target_stem)

    if not candidates:
        # Fallback: check if the target exactly matches a node path
        # (useful if the database has other keys or target is already relative/absolute path)
        with graph_store.get_conn() as conn:
            cursor = conn.execute("SELECT path, mtime FROM nodes WHERE path LIKE ?", (f"%{target_normalized}%",))
            candidates = [dict(row) for row in cursor]

    if not candidates:
        return [], False

    # Filter candidates by exact path/suffix if folder is specified
    if has_folder:
        matched = []
        for cand in candidates:
            cand_path = cand["path"].replace("\\", "/")
            if cand_path.endswith(target_normalized) or cand_path.endswith(target_normalized + ".md"):
                matched.append(cand)
        if matched:
            return [m["path"] for m in matched], len(matched) > 1

    # 2. Prefer a note in the same folder as the linking note
    source_dir = Path(source_path).parent.resolve()
    same_dir_candidates = []
    for cand in candidates:
        cand_dir = Path(cand["path"]).parent.resolve()
        if cand_dir == source_dir:
            same_dir_candidates.append(cand)
    
    if len(same_dir_candidates) == 1:
        return [same_dir_candidates[0]["path"]], False
    elif len(same_dir_candidates) > 1:
        # If multiple in same dir (unlikely but possible if extension differs),
        # fall back to mtime on same_dir_candidates
        candidates = same_dir_candidates

    # 3. Prefer the most recently modified candidate (highest mtime)
    if len(candidates) > 1:
        candidates_sorted = sorted(candidates, key=lambda x: x["mtime"], reverse=True)
        newest_mtime = candidates_sorted[0]["mtime"]
        # Check if there's a tie for newest mtime
        ties = [c for c in candidates_sorted if c["mtime"] == newest_mtime]
        if len(ties) == 1:
            return [ties[0]["path"]], False
        # If there is a tie, they remain ambiguous
        return [t["path"] for t in ties], True

    # Single candidate found
    return [candidates[0]["path"]], False
