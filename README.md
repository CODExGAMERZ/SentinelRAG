# 🛡️ SentinelRAG

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Local First](https://img.shields.io/badge/architecture-local--first-green.svg)](#)
[![Ollama Supported](https://img.shields.io/badge/ollama-supported-orange.svg)](https://ollama.com/)

**SentinelRAG** is a high-performance, local-first Retrieval-Augmented Generation (RAG) CLI tool designed to search your local files and answer queries about them with complete privacy. 

It is engineered to run seamlessly on consumer-grade hardware. It uses a lightweight embedded database tier, local token hashing and cosine similarity for vector indexing, and integrates directly with **Ollama** for local LLM generation. When Ollama is offline, it gracefully falls back to a cited extractive answer mode to prevent service failure.

SentinelRAG was created by **CODExGAMERZ (Aryan)**.

---

## ✨ Features

- 📁 **Multi-format Ingestion**: Indexes `.md`, `.txt`, `.csv`, `.html`, `.htm`, and optional `.pdf` files.
- ⚡ **Extremely Lightweight**: Operates with zero background daemons or Docker container dependencies.
- 🧠 **Hybrid Search Context**: Complements semantic vector matching with lightweight entity-style graph relationship context.
- ⚙️ **Hardware-Adaptive Profiling**: Automatically detects CPU cores, RAM, and available GPU constraints to recommend the best local model profile and thread concurrency settings.
- 🚀 **Ollama Helper CLI**: Checks for local Ollama installations, writes configuration defaults, and pulls recommended models directly from the command line.
- 🔍 **Device-Wide Scanning**: Recursively indexes entire directories or common user folders (Desktop, Documents, GitHub, etc.) while automatically skipping system noise, caches, venvs, and sensitive credential files.
- 💬 **Grounded Answers**: Always generates responses backed by local file citations and relevance scores.

---

## 🛠️ Installation

### Option 1: Global CLI Install via `pipx` (Recommended)

To run `sentinelrag` globally from any terminal on your system, install it using `pipx`:

```bash
# Install pipx if you don't have it
pip install pipx
pipx ensurepath

# Install SentinelRAG directly from the GitHub repository
pipx install git+https://github.com/CODExGAMERZ/SentinelRAG.git
```

> [!NOTE]
> **Windows Environment Variables**: If the command is not recognized after installation, ensure that the Python user Scripts directory is added to your system `PATH`.
> Typical path: `%APPDATA%\Python\Python314\Scripts` (or equivalent for your Python version).

---

### Option 2: Local Development Install

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/CODExGAMERZ/SentinelRAG.git
cd SentinelRAG
python -m pip install -e .
```

#### Install Optional Dependency Extras:

```bash
# For development dependencies (pytest)
python -m pip install -e ".[dev]"

# For PDF ingestion support
python -m pip install -e ".[pdf]"

# For advanced optional dependencies (Qdrant Edge, FalkorDBLite, nvidia-ml-py)
python -m pip install -e ".[edge]"
```

---

## 🚀 Quick Start Guide

If you are running from the cloned repository without a global installation:

```powershell
# Windows PowerShell example
$env:PYTHONPATH='src'
python -m sentinelrag profile
python -m sentinelrag setup-ollama
python -m sentinelrag ingest README.md --reset
python -m sentinelrag ask "What does this project say about hardware profiling?"
```

If you installed the package globally:

```bash
sentinelrag profile
sentinelrag setup-ollama
sentinelrag index-pc --dry-run
sentinelrag ask "What does SentinelRAG do?"
```

---

## 💻 Command Reference

### `sentinelrag profile`
Profiles your local machine's system resources and calculates optimal model parameters.
- Inspects operating system, CPU cores, total RAM, and disk space.
- Detects GPU hardware availability (such as NVIDIA GeForce/RTX cards) and sets safe VRAM model allocation budgets.
- Recommends the corresponding hardware tier, Ollama model name, context size (`num_ctx`), and `OLLAMA_NUM_PARALLEL` concurrency limits.

---

### `sentinelrag setup-ollama`
Sets up and configures the connection to your local Ollama instance.
- Detects whether Ollama is installed and running on your system.
- Recommends a model size based on hardware:
  - **Low/Entry Tier**: `qwen2.5:3b`
  - **Mid-Range Tier**: `qwen2.5:7b`
  - **High-End Tier**: `qwen2.5:14b`
- Updates the local configuration file and automatically pulls the model if specified.

**Examples:**
```bash
# Inspect status and recommend settings
sentinelrag setup-ollama

# Install Ollama if missing
sentinelrag setup-ollama --install

# Install Ollama and pull the recommended model automatically
sentinelrag setup-ollama --install --pull-model

# Specify a custom model and pull it
sentinelrag setup-ollama --model qwen2.5:7b --pull-model
```

---

### `sentinelrag ingest`
Reads and chunks files or directories to index them into your local storage.

```bash
# Ingest a folder
sentinelrag ingest ./docs

# Ingest a specific file and clear previous indexes for that file
sentinelrag ingest README.md --reset

# Ingest into a specific namespace collection
sentinelrag ingest ./docs --collection research
```

Supported extensions: `.md`, `.txt`, `.csv`, `.html`, `.htm`, `.pdf` (with `pypdf` installed), and `.env` (only if explicitly pointed to).

---

### `sentinelrag index-pc`
Automatically scans and indexes supported files from common user folders to build a personal knowledge base across your device.
- **Scanned Paths**: Desktop, Documents, Downloads, GitHub, OneDrive, Pictures, Videos, and the User Home directory.
- **Default Skips**: System directories, temporary caches, virtual environments (`.venv`), `node_modules`, folders over 25 MB, and files containing potential secrets (`.env`, `token`, `password`, `key`, etc.).

```bash
# Preview what files would be scanned without indexing
sentinelrag index-pc --dry-run

# Start a full PC indexing run, resetting any previous index
sentinelrag index-pc --reset

# Limit the scan to a custom root directory
sentinelrag index-pc --root ~/Documents --root ~/GitHub

# Include files that might contain sensitive keywords or configurations
sentinelrag index-pc --include-sensitive
```

---

### `sentinelrag ask`
Queries the indexed vector and graph stores.

```bash
# Ask a general question
sentinelrag ask "What does this project say about hardware profiling?"

# Ask with a custom retrieval depth
sentinelrag ask "What is the storage strategy?" --top-k 5

# Ask inside a custom collection and return raw JSON outputs
sentinelrag ask "What is the storage strategy?" --collection research --json
```

> [!TIP]
> **Extractive Fallback**: If Ollama is offline or unavailable, SentinelRAG automatically enters extractive fallback mode, returning direct cited text segments matching the query instead of throwing an error.

---

### `sentinelrag doctor`
Diagnostic utility displaying the health and location of all system components.
- Logs the shared global application storage paths.
- Verifies the state of config files and checks which optional dependencies (`qdrant-edge-py`, `falkordblite`, `graphiti`, `langgraph`, `nvidia-ml-py`) are available.
- Reports indexed item counts (total chunks and graph facts).
- Displays Ollama connection status, server version, and installed model lists.

```bash
# Print diagnostic summary
sentinelrag doctor

# Print diagnostics in JSON format for scripting
sentinelrag doctor --json
```

---

## 💾 Storage & Configuration

### Storage Location
To make the CLI accessible from any directory, SentinelRAG uses a shared global directory rather than writing index files inside your active folder:
- **Windows**: `%LOCALAPPDATA%\SentinelRAG`
- **macOS/Linux**: `~/.sentinelrag`

> [!TIP]
> You can override the default location by setting the `SENTINELRAG_HOME` environment variable:
> `SENTINELRAG_HOME=/custom/path sentinelrag doctor`

### Config Format
A `config.json` file is automatically created in your storage directory:
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

---

## 🔍 How it Works Under the Hood

1. **Document Parsing**: Files are parsed, cleaned, and split into smaller chunks.
2. **Deterministic Embeddings**: Text chunks are embedded locally using a lightweight in-process vectorizer.
3. **In-Process Vector Store**: Chunk coordinates and source metadata are written atomically to a local JSON-backed database.
4. **Lightweight Graph Memory**: Important capitalized entities and cross-mentions are extracted and organized into a local JSON relation graph.
5. **Context Augmentation**: The `ask` command performs a hybrid search, fetching matching vector chunks and expanding structural entities.
6. **Inference Execution**: If Ollama is reachable, the query and context are dispatched to the local LLM. If offline, the extractive matching engine presents the most relevant passages directly.

---

## 🛠️ Verification & Checks

To verify code syntax and compile the python files:

```bash
python -m compileall src
```
