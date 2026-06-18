# SentinelRAG

SentinelRAG is a local-first RAG CLI for searching your own files and asking questions over them.

It is built to be simple to run on a normal machine: local storage, local indexing, Ollama for generation when available, and a fallback extractive mode when Ollama is offline.

SentinelRAG was made by CODExGAMERZ (Aryan).

## What it does

- Indexes local files such as `.md`, `.txt`, `.csv`, `.html`, and optional `.pdf`
- Builds a local searchable knowledge base
- Adds lightweight entity-style graph context
- Answers with citations
- Can scan common user folders across your PC
- Detects Ollama, recommends a suitable local model, and can install or pull it

## Current state

This is a working MVP, not the full long-term architecture yet.

What exists today:

- `sentinelrag` CLI
- global shared storage, so the same index works from any folder
- `profile`, `setup-ollama`, `ingest`, `index-pc`, `ask`, and `doctor`
- hardware profiling
- Ollama setup guidance and model recommendation
- JSON-backed local storage
- extractive fallback answers when Ollama is unavailable

What is not in yet:

- Qdrant Edge
- FalkorDBLite
- Graphiti
- LangGraph agent workflow
- advanced scheduling and benchmarking

## Install

### Best option for other users: `pipx`

If someone gets this from GitHub and wants `sentinelrag` to work from anywhere on their machine, this is the cleanest install path:

```bash
pip install pipx
pipx ensurepath
pipx install git+https://github.com/CODExGAMERZ/SentinelRAG.git
```

If they already cloned the repo:

```bash
cd SentinelRAG
pipx install .
```

After that, they should be able to run:

```bash
sentinelrag doctor
```

from any directory.

### Local development install

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[pdf]"
python -m pip install -e ".[edge]"
```

### Windows note

If `sentinelrag` is installed but not recognized, the Python user Scripts directory is probably not on `PATH`.

Typical path:

```text
%APPDATA%\Python\Python314\Scripts
```

## Quick start

From the repo without installing:

```powershell
$env:PYTHONPATH='src'
python -m sentinelrag profile
python -m sentinelrag setup-ollama
python -m sentinelrag ingest README.md --reset
python -m sentinelrag ask "What does this project say about hardware profiling?"
```

After installation:

```bash
sentinelrag profile
sentinelrag setup-ollama
sentinelrag index-pc --dry-run
sentinelrag ask "What does SentinelRAG do?"
```

## Commands

### `sentinelrag profile`

Checks the machine and recommends safe runtime settings.

It reports things like:

- RAM and CPU
- GPU availability when detectable
- recommended hardware tier
- recommended Ollama model
- `num_ctx`
- `OLLAMA_NUM_PARALLEL`

### `sentinelrag setup-ollama`

This is the command that makes local model setup practical.

It:

- checks whether Ollama is installed
- recommends a model for the current machine
- writes the selected model and runtime defaults into the SentinelRAG config
- can install Ollama if missing
- can pull the recommended model

Examples:

```bash
sentinelrag setup-ollama
sentinelrag setup-ollama --install
sentinelrag setup-ollama --install --pull-model
sentinelrag setup-ollama --model qwen2.5:7b --pull-model
```

Current default mapping:

- `low` / `entry`: `qwen2.5:3b`
- `mid`: `qwen2.5:7b`
- `high`: `qwen2.5:14b`

### `sentinelrag ingest`

Indexes a file or directory you point it at.

```bash
sentinelrag ingest ./docs
sentinelrag ingest README.md --reset
sentinelrag ingest ./docs --collection research
```

Supported inputs:

- `.md`
- `.txt`
- `.csv`
- `.html`
- `.htm`
- `.pdf` with `pypdf`
- `.env` only when explicitly ingested or when you opt into sensitive files during PC indexing

### `sentinelrag index-pc`

Indexes supported files from common user folders so the tool has knowledge across the device instead of just one project.

```bash
sentinelrag index-pc --dry-run
sentinelrag index-pc --reset
sentinelrag index-pc --root ~/Documents --root ~/GitHub
sentinelrag index-pc --limit 500
```

By default it scans normal user locations such as Desktop, Documents, Downloads, GitHub, OneDrive, Pictures, Videos, and the home folder if present.

By default it skips:

- system folders
- caches
- venvs
- `node_modules`
- hidden metadata folders
- large files over 25 MB
- likely secret files like `.env`, token, password, credential, and private key files

Use this to preview first:

```bash
sentinelrag index-pc --dry-run
```

Use this only if you explicitly want likely secret files included:

```bash
sentinelrag index-pc --include-sensitive
```

### `sentinelrag ask`

Queries the indexed data.

```bash
sentinelrag ask "What does this project say about hardware profiling?"
sentinelrag ask "What is the storage strategy?" --top-k 5
sentinelrag ask "What is the storage strategy?" --collection research --json
```

Behavior:

- If Ollama is available, SentinelRAG sends retrieved evidence to the local model.
- If Ollama is unavailable, SentinelRAG returns an extractive cited answer from the best matches.

### `sentinelrag doctor`

Shows the current runtime state.

```bash
sentinelrag doctor
sentinelrag doctor --json
```

It includes:

- global storage location
- whether config exists
- current vector and graph backend
- indexed chunk and fact counts
- whether Ollama is installed
- recommended Ollama model
- Ollama API status
- available Ollama models when reachable

## Global storage

SentinelRAG uses a shared app directory instead of storing its index in the current project folder.

Windows default:

```text
%LOCALAPPDATA%\SentinelRAG
```

macOS/Linux default:

```text
~/.sentinelrag
```

You can override it:

```bash
SENTINELRAG_HOME=/path/to/storage sentinelrag doctor
```

That shared storage is what makes the CLI usable from anywhere on the machine.

## Config

SentinelRAG creates a config automatically.

Default shape:

```json
{
  "model": {
    "provider": "ollama",
    "name": "auto",
    "num_ctx": 4096,
    "num_parallel": 1
  },
  "storage": {
    "base_dir": "auto",
    "collection": "default"
  },
  "retrieval": {
    "top_k": 8,
    "graph_expansion_depth": 1
  }
}
```

Running `setup-ollama` will fill in the recommended model settings for the current device.

## How it works

1. Files are read and chunked into smaller pieces.
2. Chunks get stable IDs and metadata.
3. A deterministic local embedder creates vector-like representations.
4. The chunks are stored in local JSON-backed vector storage.
5. Simple entities are extracted into local graph-style memory.
6. `ask` retrieves relevant chunks and facts.
7. Ollama generates a grounded answer if available. Otherwise SentinelRAG returns an extractive cited answer.

## Check that it works

From any directory:

```bash
sentinelrag doctor
sentinelrag setup-ollama
sentinelrag index-pc --dry-run
```

Then run a real test:

```bash
sentinelrag index-pc --root <some-folder> --reset
sentinelrag ask "What does SentinelRAG do?"
```

If Ollama is running, answers should improve. If Ollama is offline, the command should still return a cited extractive answer instead of crashing.

## Whole PC knowledge

SentinelRAG only knows about files that have been indexed. It does not automatically read the whole machine in the background.

If you want whole-PC coverage, you need to run an indexing pass first.

Preview what would be scanned:

```bash
sentinelrag index-pc --dry-run
```

Index the normal user areas on the machine:

```bash
sentinelrag index-pc --reset
```

If you want to push it closer to full-disk coverage, give it an explicit root:

```bash
sentinelrag index-pc --root C:\ --reset
```

If you also want likely secret or config files included:

```bash
sentinelrag index-pc --root C:\ --reset --include-sensitive
```

Important limits:

- SentinelRAG only indexes supported text-like files such as `.md`, `.txt`, `.csv`, `.html`, optional `.pdf`, and explicitly allowed `.env` files.
- It still skips many machine-noise directories by design, such as caches, virtual environments, `node_modules`, and system-heavy folders.
- Large binaries, installers, and unsupported file types are not useful RAG input and are not indexed.

## Troubleshooting

### Ollama is installed but not answering

Check state:

```bash
sentinelrag setup-ollama
sentinelrag doctor
```

If Ollama is missing:

```bash
sentinelrag setup-ollama --install --pull-model
```

If it is installed but not serving, start it and check again:

```bash
ollama serve
sentinelrag doctor
```

### `sentinelrag` is not recognized

The install likely succeeded but the Scripts directory is not on `PATH`. Open a new terminal first. On Windows, check the Python user Scripts path noted above.

### No files were indexed

Run:

```bash
sentinelrag index-pc --dry-run
```

and verify the folders and file types you care about are actually being picked up.

### PDF ingestion fails

Install the PDF extra:

```bash
python -m pip install -e ".[pdf]"
```

## Development checks

```bash
python -m compileall src
```
