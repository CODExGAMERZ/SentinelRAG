from __future__ import annotations

import os
from pathlib import Path

from .ingest import is_supported_file

DEFAULT_SKIP_DIRS = {
    "$Recycle.Bin",
    ".cache",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".sentinelrag",
    ".svn",
    ".venv",
    "__pycache__",
    "AppData",
    "Cache",
    "Caches",
    "Library",
    "Microsoft",
    "node_modules",
    "Program Files",
    "Program Files (x86)",
    "ProgramData",
    "site-packages",
    "venv",
    "Windows",
}

SECRET_NAME_PARTS = {
    ".env",
    "apikey",
    "api_key",
    "credential",
    "credentials",
    "id_rsa",
    "id_ed25519",
    "password",
    "secret",
    "token",
}


def default_pc_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home / "GitHub",
        home / "OneDrive",
        home / "Pictures",
        home / "Videos",
        home,
    ]
    seen: set[Path] = set()
    roots: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in seen:
                roots.append(resolved)
                seen.add(resolved)
    return roots


def discover_pc_files(
    roots: list[Path] | None = None,
    *,
    include_sensitive: bool = False,
    max_file_mb: int = 25,
    limit: int | None = None,
) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots or default_pc_roots():
        for current_root, dir_names, file_names in os.walk(root):
            current = Path(current_root)
            dir_names[:] = [
                name
                for name in dir_names
                if _should_visit_dir(current / name)
            ]
            for file_name in file_names:
                path = current / file_name
                if _should_index_file(path, include_sensitive=include_sensitive, max_file_mb=max_file_mb):
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    found.append(resolved)
                    seen.add(resolved)
                    if limit is not None and len(found) >= limit:
                        return sorted(found)
    return sorted(found)


def iter_pc_files(
    roots: list[Path] | None = None,
    *,
    include_sensitive: bool = False,
    max_file_mb: int = 25,
    limit: int | None = None,
):
    seen: set[Path] = set()
    yielded = 0
    for root in roots or default_pc_roots():
        for current_root, dir_names, file_names in os.walk(root):
            current = Path(current_root)
            dir_names[:] = [name for name in dir_names if _should_visit_dir(current / name)]
            for file_name in file_names:
                path = current / file_name
                if not _should_index_file(path, include_sensitive=include_sensitive, max_file_mb=max_file_mb):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def _should_visit_dir(path: Path) -> bool:
    name = path.name
    if name in DEFAULT_SKIP_DIRS:
        return False
    if name.endswith(".egg-info") or name.endswith(".dist-info"):
        return False
    if name.startswith(".") and name not in {".config"}:
        return False
    return True


def _should_index_file(path: Path, *, include_sensitive: bool, max_file_mb: int) -> bool:
    if not is_supported_file(path):
        return False
    if not include_sensitive and _looks_sensitive(path):
        return False
    try:
        if path.stat().st_size > max_file_mb * 1024 * 1024:
            return False
    except OSError:
        return False
    return True


def _looks_sensitive(path: Path) -> bool:
    lowered = path.name.lower()
    return any(part in lowered for part in SECRET_NAME_PARTS)
