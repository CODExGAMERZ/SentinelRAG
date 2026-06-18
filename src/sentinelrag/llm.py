from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class OllamaStatus:
    available: bool
    message: str
    models: list[str]


@dataclass(slots=True)
class OllamaInstallPlan:
    installed: bool
    command_path: str | None
    install_command: list[str] | None
    start_command: list[str] | None
    platform_name: str


def ollama_command_path() -> str | None:
    return shutil.which("ollama")


def ollama_install_plan() -> OllamaInstallPlan:
    command_path = ollama_command_path()
    system = platform.system()
    install_command: list[str] | None = None
    start_command: list[str] | None = None

    if system == "Windows":
        if shutil.which("winget"):
            install_command = ["winget", "install", "--id", "Ollama.Ollama", "-e", "--accept-package-agreements", "--accept-source-agreements"]
        start_command = ["ollama", "serve"]
    elif system == "Darwin":
        if shutil.which("brew"):
            install_command = ["brew", "install", "ollama"]
        start_command = ["ollama", "serve"]
    elif system == "Linux":
        start_command = ["ollama", "serve"]

    return OllamaInstallPlan(
        installed=command_path is not None,
        command_path=command_path,
        install_command=install_command,
        start_command=start_command,
        platform_name=system,
    )


def ollama_status() -> OllamaStatus:
    try:
        request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        return OllamaStatus(True, "Ollama API is reachable.", models)
    except Exception as exc:
        return OllamaStatus(False, f"Ollama API is not reachable: {exc}", [])


def choose_model(configured: str, available: list[str]) -> str:
    if configured != "auto":
        return configured
    if available:
        return available[0]
    return "llama3.2:3b"


def ensure_ollama_installed() -> tuple[bool, str]:
    plan = ollama_install_plan()
    if plan.installed:
        return True, f"Ollama is installed at {plan.command_path}."
    if not plan.install_command:
        return False, f"No safe automatic Ollama install command is configured for {plan.platform_name}."
    try:
        result = subprocess.run(
            plan.install_command,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except Exception as exc:
        return False, f"Ollama install failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, f"Ollama install failed: {detail}"
    return True, "Ollama installation completed."


def pull_ollama_model(model: str) -> tuple[bool, str]:
    command = ollama_command_path()
    if not command:
        return False, "Ollama command is not installed."
    try:
        result = subprocess.run(
            [command, "pull", model],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except Exception as exc:
        return False, f"Model pull failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, f"Model pull failed: {detail}"
    return True, f"Pulled model {model}."


def generate_with_ollama(prompt: str, model: str, num_ctx: int, num_parallel: int) -> str:
    os.environ["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("response", "").strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama generation failed: HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama generation failed: {exc}") from exc
