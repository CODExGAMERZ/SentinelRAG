from __future__ import annotations

import re
from pathlib import Path


COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_collection_name(name: str) -> str:
    if not COLLECTION_RE.fullmatch(name):
        raise ValueError(
            "Invalid collection name. Use 1-128 characters from letters, numbers, dot, underscore, or hyphen."
        )
    return name


def ensure_within_base(base: Path, target: Path) -> Path:
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"Resolved path escapes base directory: {resolved_target}") from exc
    return resolved_target
