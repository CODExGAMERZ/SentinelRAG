from __future__ import annotations

import json
import os
import secrets
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
    qdrant_dirname: str = "qdrant_db"
    sqlite_filename: str = "sentinelrag.db"


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 8
    graph_expansion_depth: int = 1
    parser_tier: str = "auto"
    temporal_decay_days: int = 90
    centrality_cache_window_seconds: int = 30


@dataclass(slots=True)
class WatcherConfig:
    enabled: bool = True
    debounce_ms: int = 500
    startup_sync: bool = True


@dataclass(slots=True)
class HardwareRuntimeConfig:
    tier: str = "auto"
    parser_locked_tier: str = "auto"
    workflow_topology: str = "auto"
    allow_concurrent_llm: bool = False


@dataclass(slots=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 7419
    persist_token: bool = False
    token_file: str = "credentials"


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    hardware: HardwareRuntimeConfig = field(default_factory=HardwareRuntimeConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

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
    for child in ("qdrant", "falkor", "models", "state"):
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
    watcher = {**defaults["watcher"], **raw.get("watcher", {})}
    hardware = {**defaults["hardware"], **raw.get("hardware", {})}
    api = {**defaults["api"], **raw.get("api", {})}
    loaded = AppConfig(
        model=ModelConfig(**model),
        storage=StorageConfig(**storage),
        retrieval=RetrievalConfig(**retrieval),
        watcher=WatcherConfig(**watcher),
        hardware=HardwareRuntimeConfig(**hardware),
        api=ApiConfig(**api),
    )
    ensure_app_dirs(loaded, cwd)
    return loaded


def save_config(config: AppConfig, cwd: Path | None = None) -> None:
    base = ensure_app_dirs(config, cwd)
    path = base / "config.json"
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def credentials_path(config: AppConfig, cwd: Path | None = None) -> Path:
    return ensure_app_dirs(config, cwd) / config.api.token_file


def load_or_create_api_token(config: AppConfig, cwd: Path | None = None) -> str:
    path = credentials_path(config, cwd)
    if config.api.persist_token and path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token
