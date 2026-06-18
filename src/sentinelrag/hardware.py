from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


BYTES_PER_GB = 1024**3


@dataclass(slots=True)
class GpuInfo:
    vendor: str
    name: str
    total_gb: float | None
    free_gb: float | None
    available: bool


@dataclass(slots=True)
class HardwareProfile:
    os: str
    machine: str
    cpu_cores: int
    total_ram_gb: float
    free_ram_gb: float
    disk_free_gb: float
    gpus: list[GpuInfo]
    apple_mps: bool
    rocm: bool
    recommended_tier: str
    recommended_model: str
    recommended_ollama_model: str
    allowed_formats: list[str]
    num_ctx: int
    ollama_num_parallel: int
    safe_model_budget_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_info() -> tuple[float, float]:
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / BYTES_PER_GB, status.ullAvailPhys / BYTES_PER_GB
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return (page_size * phys_pages) / BYTES_PER_GB, (page_size * avail_pages) / BYTES_PER_GB
    except Exception:
        return 0.0, 0.0


def _nvidia_gpus() -> list[GpuInfo]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        gpus: list[GpuInfo] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuInfo(
                    vendor="nvidia",
                    name=str(name),
                    total_gb=round(memory.total / BYTES_PER_GB, 2),
                    free_gb=round(memory.free / BYTES_PER_GB, 2),
                    available=True,
                )
            )
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _rocm_available() -> bool:
    if shutil.which("rocm-smi"):
        return True
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _apple_mps_available() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _allowed_formats(has_nvidia: bool, apple_mps: bool) -> list[str]:
    if apple_mps or not has_nvidia:
        return ["GGUF"]
    if platform.system() == "Linux" and has_nvidia:
        return ["GGUF", "AWQ", "GPTQ"]
    return ["GGUF"]


def _recommend(total_ram_gb: float, free_ram_gb: float, max_vram_gb: float) -> tuple[str, str, str, int, int, float]:
    reserved_os = max(2.0, total_ram_gb * 0.2)
    framework_overhead = 0.5
    usable_ram = max(0.5, free_ram_gb - reserved_os - framework_overhead)
    accelerator_budget = max_vram_gb * 0.85 if max_vram_gb else 0.0
    safe_budget = max(usable_ram, accelerator_budget)

    if total_ram_gb < 8 and max_vram_gb < 4:
        return "low", "1B-3B GGUF (Phi-3 Mini or Qwen2.5 3B)", "qwen2.5:3b", 2048, 1, round(safe_budget, 2)
    if total_ram_gb < 16 and max_vram_gb < 6:
        return "entry", "3B-4B GGUF", "qwen2.5:3b", 4096, 1, round(safe_budget, 2)
    if total_ram_gb < 32 and max_vram_gb < 12:
        parallel = 2 if safe_budget >= 10 else 1
        return "mid", "7B-14B GGUF/quantized", "qwen2.5:7b", 8192, parallel, round(safe_budget, 2)
    return "high", "14B-30B quantized or MoE", "qwen2.5:14b", 8192, 2, round(safe_budget, 2)


def detect_hardware(cwd: Path | None = None) -> HardwareProfile:
    total_ram, free_ram = _ram_info()
    disk = shutil.disk_usage(cwd or Path.cwd())
    gpus = _nvidia_gpus()
    rocm = _rocm_available()
    apple_mps = _apple_mps_available()
    max_vram = max((gpu.free_gb or 0.0 for gpu in gpus), default=0.0)
    tier, model, ollama_model, num_ctx, parallel, budget = _recommend(total_ram, free_ram, max_vram)

    return HardwareProfile(
        os=platform.system(),
        machine=platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        total_ram_gb=round(total_ram, 2),
        free_ram_gb=round(free_ram, 2),
        disk_free_gb=round(disk.free / BYTES_PER_GB, 2),
        gpus=gpus,
        apple_mps=apple_mps,
        rocm=rocm,
        recommended_tier=tier,
        recommended_model=model,
        recommended_ollama_model=ollama_model,
        allowed_formats=_allowed_formats(any(g.vendor == "nvidia" for g in gpus), apple_mps),
        num_ctx=num_ctx,
        ollama_num_parallel=parallel,
        safe_model_budget_gb=budget,
    )


def profile_json(profile: HardwareProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2)
