from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    provider: str = "ollama"
    name: str = "auto"
    num_ctx: int = 4096
    num_parallel: int = 1


@dataclass(slots=True)
class StorageConfig:
    base_dir: str = "auto"
    collection: str = "default"


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 8
    graph_expansion_depth: int = 1


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_app_dir() -> Path:
    override = os.environ.get("SENTINELRAG_HOME")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "SentinelRAG"
    return Path.home() / ".sentinelrag"


def resolve_base_dir(base_dir: str, cwd: Path | None = None) -> Path:
    if base_dir == "auto":
        return default_app_dir()
    path = Path(base_dir).expanduser()
    if path.is_absolute():
        return path
    return (cwd or Path.cwd()) / path


def app_dir(cwd: Path | None = None) -> Path:
    return default_app_dir()


def config_path(cwd: Path | None = None) -> Path:
    return app_dir(cwd) / "config.json"


def ensure_app_dirs(config: AppConfig, cwd: Path | None = None) -> Path:
    base = resolve_base_dir(config.storage.base_dir, cwd)
    for child in ("qdrant", "falkor", "models"):
        (base / child).mkdir(parents=True, exist_ok=True)
    return base


def load_config(cwd: Path | None = None) -> AppConfig:
    path = config_path(cwd)
    config = AppConfig()
    if not path.exists():
        ensure_app_dirs(config, cwd)
        save_config(config, cwd)
        return config

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        ensure_app_dirs(config, cwd)
        save_config(config, cwd)
        return config
    defaults = config.to_dict()
    model = {**defaults["model"], **raw.get("model", {})}
    storage = {**defaults["storage"], **raw.get("storage", {})}
    retrieval = {**defaults["retrieval"], **raw.get("retrieval", {})}
    loaded = AppConfig(
        model=ModelConfig(**model),
        storage=StorageConfig(**storage),
        retrieval=RetrievalConfig(**retrieval),
    )
    ensure_app_dirs(loaded, cwd)
    return loaded


def save_config(config: AppConfig, cwd: Path | None = None) -> None:
    base = ensure_app_dirs(config, cwd)
    path = base / "config.json"
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
