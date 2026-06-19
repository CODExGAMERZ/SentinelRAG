from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

from .agents.workflow import select_workflow_topology
from .api import run_api_server
from .config import ensure_app_dirs, load_config, save_config
from .graph.graph_store import GraphStore
from .hardware_profiler import detect_hardware
from .llm import ensure_ollama_installed, ollama_install_plan, ollama_status, pull_ollama_model
from .obsidian.parser import probe_parser_tier
from .obsidian.watcher import VaultWatcher
from .paths import validate_collection_name
from .rag import ask_question, format_answer, result_json
from .storage.vector_store import VectorStore


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_profile(args: argparse.Namespace) -> int:
    config = load_config()
    profile = detect_hardware()
    parser_probe = probe_parser_tier()
    topology = select_workflow_topology(profile.recommended_tier)
    config.model.num_ctx = profile.num_ctx
    config.model.num_parallel = profile.ollama_num_parallel
    config.hardware.tier = profile.recommended_tier
    config.hardware.workflow_topology = profile.agent_topology
    config.hardware.allow_concurrent_llm = profile.allow_concurrent_llm
    config.hardware.parser_locked_tier = parser_probe.tier
    config.retrieval.parser_tier = parser_probe.tier
    save_config(config)
    
    if args.json:
        payload = profile.to_dict()
        payload["parser_tier"] = parser_probe.tier
        payload["parser_engine"] = parser_probe.engine
        payload["workflow_nodes"] = list(topology.nodes)
        print(json.dumps(payload, indent=2))
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
        print(f"Parser tier: {parser_probe.tier} ({parser_probe.engine})")
        print(f"Workflow topology: {' -> '.join(topology.nodes)}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    vault_path = Path(args.path)
    if not vault_path.exists():
        print(f"Error: path '{args.path}' does not exist.", file=sys.stderr)
        return 1

    config = load_config()
    collection = validate_collection_name(args.collection or config.storage.collection)
    base = ensure_app_dirs(config)
    
    vector_store = VectorStore(base, collection)
    graph_store = GraphStore(base, config.storage.sqlite_filename)
    
    try:
        if args.reset:
            vector_store.reset()
            graph_store.reset()

        # Initialize the Watcher which does the startup sync
        watcher = VaultWatcher(vault_path, config, vector_store, graph_store)
        
        try:
            watcher.sync_all(force=args.force)
        except Exception as exc:
            print(f"Startup sync failed: {exc}", file=sys.stderr)
            return 1

        if args.watch:
            print(f"Watching '{vault_path}' for changes. Press Ctrl+C to stop...")
            watcher.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping watcher...")
                watcher.stop()
    finally:
        vector_store.close()
            
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not args.question or not args.question.strip():
        print("Error: question is empty.", file=sys.stderr)
        return 1
    config = load_config()
    collection = validate_collection_name(args.collection or config.storage.collection)
    result = ask_question(args.question, config, collection=collection, top_k=args.top_k)
    print(result_json(result) if args.json else format_answer(result))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config()
    base = ensure_app_dirs(config)
    vector_store = VectorStore(base, config.storage.collection)
    graph_store = GraphStore(base, config.storage.sqlite_filename)
    
    try:
        hardware = detect_hardware()
        parser_probe = probe_parser_tier()
        install_plan = ollama_install_plan()
        status = ollama_status()
        topology = select_workflow_topology(hardware.recommended_tier)
        
        report = {
            "storage_dir": str(base.resolve()),
            "config_exists": (base / "config.json").exists(),
            "dependencies": {
                "qdrant-client": importlib.util.find_spec("qdrant_client") is not None,
                "langgraph": importlib.util.find_spec("langgraph") is not None,
                "watchdog": importlib.util.find_spec("watchdog") is not None,
                "markdown-it-py": importlib.util.find_spec("markdown_it") is not None,
            },
            "vector_backend": vector_store.backend,
            "vector_chunks": vector_store.count(),
            "graph_backend": graph_store.backend,
            "recommended_ollama_model": hardware.recommended_ollama_model,
            "recommended_tier": hardware.recommended_tier,
            "parser_tier": parser_probe.tier,
            "parser_engine": parser_probe.engine,
            "workflow_nodes": list(topology.nodes),
            "ollama_installed": install_plan.installed,
            "ollama_command_path": install_plan.command_path,
            "ollama_available": status.available,
            "ollama_message": status.message,
            "ollama_models": status.models,
        }
    finally:
        vector_store.close()
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    config = load_config()
    if args.port is not None:
        config.api.port = args.port
    if args.persist_token:
        config.api.persist_token = True
        save_config(config)
    print(f"Serving SentinelRAG on http://{config.api.host}:{config.api.port}")
    run_api_server(config)
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

    ingest = subparsers.add_parser("ingest", help="Ingest local Markdown Obsidian vaults.")
    ingest.add_argument("path", help="Vault directory to ingest.")
    ingest.add_argument("--reset", action="store_true", help="Reset collection before ingesting.")
    ingest.add_argument("--force", action="store_true", help="Force full re-indexing of all files.")
    ingest.add_argument("--collection", default=None, help="Collection name.")
    ingest.add_argument("--watch", action="store_true", help="Start filesystem watcher to track changes in real-time.")
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="Ask a question against indexed documents.")
    ask.add_argument("question", help="Question to answer.")
    ask.add_argument("--collection", default=None, help="Collection name.")
    ask.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    ask.set_defaults(func=cmd_ask)

    doctor = subparsers.add_parser("doctor", help="Check local runtime dependencies.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.set_defaults(func=cmd_doctor)

    serve = subparsers.add_parser("serve", help="Run the local authenticated API daemon.")
    serve.add_argument("--port", type=int, default=None, help="Override the configured port.")
    serve.add_argument("--persist-token", action="store_true", help="Persist the generated API token to disk.")
    serve.set_defaults(func=cmd_serve)
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
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        import traceback
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
