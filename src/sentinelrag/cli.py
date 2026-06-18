from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

from .config import ensure_app_dirs, load_config, save_config
from .hardware import detect_hardware, profile_json
from .ingest import discover_files, make_chunks
from .llm import ensure_ollama_installed, ollama_install_plan, ollama_status, pull_ollama_model
from .pcscan import default_pc_roots, discover_pc_files, iter_pc_files
from .paths import validate_collection_name
from .rag import ask_question, format_answer, result_json
from .storage import GraphMemory, VectorStore


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def cmd_profile(args: argparse.Namespace) -> int:
    config = load_config()
    profile = detect_hardware()
    config.model.num_ctx = profile.num_ctx
    config.model.num_parallel = profile.ollama_num_parallel
    save_config(config)
    if args.json:
        print(profile_json(profile))
    else:
        print(f"OS: {profile.os} {profile.machine}")
        print(f"CPU cores: {profile.cpu_cores}")
        print(f"RAM: {profile.free_ram_gb} GB free / {profile.total_ram_gb} GB total")
        print(f"Disk free: {profile.disk_free_gb} GB")
        print(f"GPU count: {len(profile.gpus)}")
        print(f"Recommended tier: {profile.recommended_tier}")
        print(f"Recommended model: {profile.recommended_model}")
        print(f"Recommended Ollama model: {profile.recommended_ollama_model}")
        print(f"Allowed formats: {', '.join(profile.allowed_formats)}")
        print(f"num_ctx: {profile.num_ctx}")
        print(f"OLLAMA_NUM_PARALLEL: {profile.ollama_num_parallel}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config()
    collection = validate_collection_name(args.collection or config.storage.collection)
    base = ensure_app_dirs(config)
    vector_store = VectorStore(base, collection)
    graph = GraphMemory(base, collection)
    if args.reset:
        vector_store.reset()
        graph.reset()

    files = discover_files(Path(args.path))
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    total_chunks = 0
    failures: list[str] = []
    for file in files:
        try:
            chunks = make_chunks(file)
            if chunks:
                vector_store.upsert_chunks(chunks)
                graph.upsert_chunks(chunks)
                total_chunks += len(chunks)
        except Exception as exc:
            failures.append(f"{file}: {exc}")

    print(f"Ingested {total_chunks} chunks from {len(files) - len(failures)} files into collection '{collection}'.")
    print(f"Vector backend: {vector_store.backend}; graph backend: {graph.backend}.")
    if failures:
        print("Failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    return 0


def _ingest_files(files: list[Path], collection: str, reset: bool) -> tuple[int, int, list[str], str, str]:
    config = load_config()
    base = ensure_app_dirs(config)
    vector_store = VectorStore(base, collection)
    graph = GraphMemory(base, collection)
    if reset:
        vector_store.reset()
        graph.reset()

    total_chunks = 0
    failures: list[str] = []
    for file in files:
        try:
            chunks = make_chunks(file)
            if chunks:
                vector_store.upsert_chunks(chunks)
                graph.upsert_chunks(chunks)
                total_chunks += len(chunks)
        except Exception as exc:
            failures.append(f"{file}: {exc}")
    return total_chunks, len(files) - len(failures), failures, vector_store.backend, graph.backend


def _ingest_file_stream(
    files,
    collection: str,
    reset: bool,
) -> tuple[int, int, int, list[str], str, str]:
    config = load_config()
    base = ensure_app_dirs(config)
    vector_store = VectorStore(base, collection)
    graph = GraphMemory(base, collection)
    if reset:
        vector_store.reset()
        graph.reset()

    total_chunks = 0
    indexed_files = 0
    scanned_files = 0
    failures: list[str] = []
    for file in files:
        scanned_files += 1
        try:
            chunks = make_chunks(file)
            if chunks:
                vector_store.upsert_chunks(chunks)
                graph.upsert_chunks(chunks)
                total_chunks += len(chunks)
                indexed_files += 1
        except Exception as exc:
            failures.append(f"{file}: {exc}")
    return total_chunks, indexed_files, scanned_files, failures, vector_store.backend, graph.backend


def cmd_index_pc(args: argparse.Namespace) -> int:
    config = load_config()
    collection = validate_collection_name(args.collection or config.storage.collection)
    roots = [Path(root).expanduser() for root in args.root] if args.root else default_pc_roots()
    files = discover_pc_files(
        roots,
        include_sensitive=args.include_sensitive,
        max_file_mb=args.max_file_mb,
        limit=args.limit,
    )
    if args.dry_run:
        print(f"Would index {len(files)} files into collection '{collection}'.")
        for root in roots:
            print(f"Root: {root}")
        for file in files[:50]:
            print(file)
        if len(files) > 50:
            print(f"... and {len(files) - 50} more")
        return 0
    if not files:
        print("No supported PC files found.", file=sys.stderr)
        return 1
    file_stream = iter_pc_files(
        roots,
        include_sensitive=args.include_sensitive,
        max_file_mb=args.max_file_mb,
        limit=args.limit,
    )
    total_chunks, indexed_files, scanned_files, failures, vector_backend, graph_backend = _ingest_file_stream(
        file_stream, collection, args.reset
    )
    print(f"Indexed {total_chunks} chunks from {indexed_files} PC files into collection '{collection}'.")
    print(f"Scanned {scanned_files} candidate files.")
    print(f"Vector backend: {vector_backend}; graph backend: {graph_backend}.")
    if failures:
        print("Failures:", file=sys.stderr)
        for failure in failures[:25]:
            print(f"- {failure}", file=sys.stderr)
        if len(failures) > 25:
            print(f"... and {len(failures) - 25} more failures", file=sys.stderr)
        return 2
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    config = load_config()
    collection = validate_collection_name(args.collection or config.storage.collection)
    result = ask_question(args.question, config, collection=collection, top_k=args.top_k)
    print(result_json(result) if args.json else format_answer(result))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config()
    base = ensure_app_dirs(config)
    vector_store = VectorStore(base, config.storage.collection)
    graph = GraphMemory(base, config.storage.collection)
    hardware = detect_hardware()
    install_plan = ollama_install_plan()
    status = ollama_status()
    report = {
        "storage_dir": str(base.resolve()),
        "config_exists": (base / "config.json").exists(),
        "optional_dependencies": {
            "qdrant_edge_py": importlib.util.find_spec("qdrant_edge_py") is not None
            or importlib.util.find_spec("qdrant_edge") is not None,
            "falkordblite": importlib.util.find_spec("falkordblite") is not None,
            "graphiti": importlib.util.find_spec("graphiti") is not None,
            "langgraph": importlib.util.find_spec("langgraph") is not None,
            "nvidia_ml_py": importlib.util.find_spec("pynvml") is not None,
        },
        "vector_backend": vector_store.backend,
        "vector_chunks": vector_store.count(),
        "graph_backend": graph.backend,
        "graph_facts": graph.count_facts(),
        "recommended_ollama_model": hardware.recommended_ollama_model,
        "ollama_installed": install_plan.installed,
        "ollama_command_path": install_plan.command_path,
        "ollama_install_command": install_plan.install_command,
        "ollama_available": status.available,
        "ollama_message": status.message,
        "ollama_models": status.models,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


def cmd_setup_ollama(args: argparse.Namespace) -> int:
    config = load_config()
    hardware = detect_hardware()
    plan = ollama_install_plan()
    model = args.model or hardware.recommended_ollama_model

    if args.json:
        report = {
            "ollama_installed": plan.installed,
            "ollama_command_path": plan.command_path,
            "ollama_install_command": plan.install_command,
            "recommended_ollama_model": hardware.recommended_ollama_model,
            "selected_model": model,
            "num_ctx": hardware.num_ctx,
            "num_parallel": hardware.ollama_num_parallel,
        }
        print(json.dumps(report, indent=2))
        return 0

    print(f"Recommended Ollama model: {hardware.recommended_ollama_model}")
    print(f"Selected model: {model}")
    print(f"num_ctx: {hardware.num_ctx}")
    print(f"OLLAMA_NUM_PARALLEL: {hardware.ollama_num_parallel}")

    if not plan.installed:
        if not args.install:
            print("Ollama is not installed. Re-run with --install to install it.")
            if plan.install_command:
                print(f"Install command: {' '.join(plan.install_command)}")
            return 1
        ok, message = ensure_ollama_installed()
        print(message)
        if not ok:
            return 1
    else:
        print(f"Ollama is installed at {plan.command_path}.")

    config.model.name = model
    config.model.num_ctx = hardware.num_ctx
    config.model.num_parallel = hardware.ollama_num_parallel
    save_config(config)

    if args.pull_model:
        ok, message = pull_ollama_model(model)
        print(message)
        if not ok:
            return 1
    else:
        print("Model pull skipped. Re-run with --pull-model to download it.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinelrag")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Detect hardware and recommend runtime settings.")
    profile.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    profile.set_defaults(func=cmd_profile)

    setup_ollama = subparsers.add_parser("setup-ollama", help="Detect, install, and configure Ollama plus a recommended model.")
    setup_ollama.add_argument("--install", action="store_true", help="Install Ollama if it is missing.")
    setup_ollama.add_argument("--pull-model", action="store_true", help="Pull the selected Ollama model.")
    setup_ollama.add_argument("--model", default=None, help="Override the recommended Ollama model.")
    setup_ollama.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    setup_ollama.set_defaults(func=cmd_setup_ollama)

    ingest = subparsers.add_parser("ingest", help="Ingest local documents.")
    ingest.add_argument("path", help="File or directory to ingest.")
    ingest.add_argument("--reset", action="store_true", help="Reset collection before ingesting.")
    ingest.add_argument("--collection", default=None, help="Collection name.")
    ingest.set_defaults(func=cmd_ingest)

    index_pc = subparsers.add_parser("index-pc", help="Index supported documents from normal user folders.")
    index_pc.add_argument("--root", action="append", default=[], help="Root directory to scan. Can be passed multiple times.")
    index_pc.add_argument("--reset", action="store_true", help="Reset collection before indexing.")
    index_pc.add_argument("--collection", default=None, help="Collection name.")
    index_pc.add_argument("--max-file-mb", type=int, default=25, help="Skip files larger than this size.")
    index_pc.add_argument("--limit", type=int, default=None, help="Maximum number of files to index.")
    index_pc.add_argument("--include-sensitive", action="store_true", help="Include likely secret/credential files.")
    index_pc.add_argument("--dry-run", action="store_true", help="List files that would be indexed without writing.")
    index_pc.set_defaults(func=cmd_index_pc)

    ask = subparsers.add_parser("ask", help="Ask a question against indexed documents.")
    ask.add_argument("question", help="Question to answer.")
    ask.add_argument("--collection", default=None, help="Collection name.")
    ask.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    ask.set_defaults(func=cmd_ask)

    doctor = subparsers.add_parser("doctor", help="Check local runtime dependencies.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return args.func(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
