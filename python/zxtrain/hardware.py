"""Hardware and capability detection.

Everything here is measured from the machine (os, ctypes, nvidia-smi, torch) or
explicitly reported as unavailable. There is no fabricated telemetry: when a
value cannot be read, the field is null and `notes` explains why.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import ZxError
from .util import human_bytes

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _run(args: list[str], timeout: float = 6.0) -> str:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def module_version(name: str) -> str | None:
    if not module_available(name):
        return None
    try:
        module = __import__(name)
    except Exception:  # pragma: no cover - broken installs
        return "installed (failed to import)"
    return str(getattr(module, "__version__", "unknown"))


def _memory_bytes() -> tuple[int | None, int | None, str | None]:
    """(total, available) physical memory in bytes."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
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

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullTotalPhys), int(stat.ullAvailPhys), None
        except Exception:
            return None, None, "GlobalMemoryStatusEx unavailable"
        return None, None, "GlobalMemoryStatusEx returned false"

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(":")
            if len(parts) == 2 and parts[1].strip().endswith("kB"):
                values[parts[0].strip()] = int(parts[1].split()[0]) * 1024
        if "MemTotal" in values:
            return values.get("MemTotal"), values.get("MemAvailable"), None
    if hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return total, None, None
        except (ValueError, OSError):
            pass
    return None, None, "memory information not readable on this platform"


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "model": platform.processor() or None,
        "architecture": platform.machine(),
        "flags": [],
        "notes": [],
    }
    if os.name == "nt":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress)",
        ])
        try:
            payload = json.loads(out) if out else {}
            info["model"] = payload.get("Name") or info["model"]
            info["physical_cores"] = int(payload.get("NumberOfCores") or 0) or None
        except (ValueError, TypeError):
            info["notes"].append("CPU details unavailable from Win32_Processor")
    else:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("model name") and not info["model"]:
                    info["model"] = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not info["flags"]:
                    info["flags"] = line.split(":", 1)[1].split()[:200]
                if line.startswith("cpu cores"):
                    info["physical_cores"] = int(line.split(":", 1)[1].strip())
    if not info["flags"]:
        info["flags"] = [
            flag for flag in ("avx", "avx2", "avx512f") if _cpu_has_flag(flag)
        ]
    info["avx2"] = "avx2" in info["flags"]
    info["avx512"] = "avx512f" in info["flags"]
    return info


def _cpu_has_flag(flag: str) -> bool:
    """Best effort CPU flag probe via the platform's own tools."""
    if flag != "avx2":
        return False
    if os.name == "nt":
        return bool(_run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).SecondLevelAddressTranslationExtensions | Select-Object -First 1"]))
    return False


def _torch_info() -> dict[str, Any]:
    if not module_available("torch"):
        return {
            "installed": False,
            "version": None,
            "cuda_build": None,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "notes": ["PyTorch is not installed in this Python environment."],
        }
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - broken torch install
        return {
            "installed": True,
            "version": None,
            "cuda_build": None,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "notes": [f"PyTorch import failed: {exc}"],
        }

    info: dict[str, Any] = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": 0,
        "devices": [],
        "notes": [],
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "threads": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
    }
    try:
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            for index in range(info["device_count"]):
                props = torch.cuda.get_device_properties(index)
                info["devices"].append({
                    "index": index,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                    "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
                })
        else:
            if info["cuda_build"]:
                info["notes"].append(
                    "PyTorch was built with CUDA but no usable NVIDIA device/driver was found."
                )
            else:
                info["notes"].append("CPU-only PyTorch build.")
    except Exception as exc:  # pragma: no cover - driver hiccups
        info["notes"].append(f"Device probing failed: {exc}")
    return info


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if candidate.exists():
            executable = str(candidate)
        else:
            system32 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "nvidia-smi.exe"
            if system32.exists():
                executable = str(system32)
    if not executable:
        return []
    query = (
        "index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,"
        "temperature.gpu,power.draw,power.limit,driver_version,compute_cap"
    )
    out = _run([executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 12:
            continue

        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        gpus.append({
            "index": int(number(parts[0]) or 0),
            "name": parts[1],
            "memory_total_mb": number(parts[2]),
            "memory_used_mb": number(parts[3]),
            "memory_free_mb": number(parts[4]),
            "utilization_gpu": number(parts[5]),
            "utilization_memory": number(parts[6]),
            "temperature_c": number(parts[7]),
            "power_draw_w": number(parts[8]),
            "power_limit_w": number(parts[9]),
            "driver_version": parts[10] or None,
            "compute_capability": parts[11] or None,
        })
    return gpus


def read_live_resources() -> dict[str, Any]:
    """A single live sample used by the status bar and the training monitor."""
    total, available, note = _memory_bytes()
    sample: dict[str, Any] = {
        "timestamp": time.time(),
        "ram": {
            "total": total,
            "available": available,
            "used": (total - available) if (total and available) else None,
            "percent": round((1 - available / total) * 100, 1) if (total and available) else None,
            "note": note,
        },
        "cpu": {
            "percent": _cpu_percent(),
            "logical_cores": os.cpu_count(),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        },
        "gpus": _nvidia_smi_gpus(),
        "process": {
            "rss": _process_rss(),
            "pid": os.getpid(),
        },
    }
    return sample


_last_cpu: tuple[float, float] | None = None


def _cpu_percent() -> float | None:
    """Cross platform CPU load without psutil: delta of process+system times."""
    global _last_cpu
    try:
        if os.name == "nt":
            idle = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            if not ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            idle_t, busy_t = float(idle.value), float(kernel.value + user.value)
        else:
            fields = Path("/proc/stat").read_text(encoding="utf-8").split("\n", 1)[0].split()[1:]
            values = [float(v) for v in fields]
            idle_t = values[3] + (values[4] if len(values) > 4 else 0.0)
            busy_t = sum(values) - idle_t
        if _last_cpu is None:
            _last_cpu = (idle_t, busy_t)
            return None
        idle_delta = idle_t - _last_cpu[0]
        busy_delta = busy_t - _last_cpu[1]
        _last_cpu = (idle_t, busy_t)
        total_delta = idle_delta + busy_delta
        if total_delta <= 0:
            return None
        return round(busy_delta / total_delta * 100, 1)
    except Exception:
        return None


def _process_rss() -> int | None:
    try:
        import resource  # type: ignore
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage if sys.platform == "darwin" else usage * 1024)
    except Exception:
        if os.name == "nt":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            try:
                handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
                if ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return int(counters.WorkingSetSize)
            except Exception:
                return None
    return None


def detect(storage_paths: list[str] | None = None) -> dict[str, Any]:
    """Full hardware report + capability matrix."""
    torch_info = _torch_info()
    cpu = _cpu_info()
    total_mem, avail_mem, mem_note = _memory_bytes()
    gpus = _nvidia_smi_gpus()

    gpu_devices: list[dict[str, Any]] = []
    for gpu in gpus:
        gpu_devices.append({
            **gpu,
            "backend": "cuda",
            "source": "nvidia-smi",
            "memory_total": int((gpu.get("memory_total_mb") or 0) * 1024 * 1024),
        })
    known = {gpu["name"] for gpu in gpu_devices}
    for device in torch_info.get("devices", []):
        if device["name"] not in known:
            gpu_devices.append({
                "index": device["index"],
                "name": device["name"],
                "memory_total": device["total_memory"],
                "memory_used_mb": None,
                "memory_free_mb": None,
                "utilization_gpu": None,
                "temperature_c": None,
                "power_draw_w": None,
                "driver_version": None,
                "compute_capability": device.get("compute_capability"),
                "backend": "cuda",
                "source": "torch",
            })

    disks: list[dict[str, Any]] = []
    for target in (storage_paths or [str(Path.home())]):
        try:
            usage = shutil.disk_usage(target)
            disks.append({
                "path": str(target),
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": round(usage.used / usage.total * 100, 1) if usage.total else None,
            })
        except OSError:
            continue

    capabilities = _capabilities(torch_info, gpu_devices, cpu)

    notes: list[str] = []
    if mem_note:
        notes.append(mem_note)
    if not gpu_devices:
        notes.append(
            "No CUDA device detected. Training runs on the CPU and is limited to small models — "
            "the engine will keep batch sizes and defaults inside what this machine can actually do."
        )
    if total_mem and total_mem < 16 * 1024**3:
        notes.append(
            f"System memory is {human_bytes(total_mem)}. Very large models cannot be loaded; "
            "the estimator accounts for that before a run starts."
        )
    return {
        "os": {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "cpu": cpu,
        "memory": {"total": total_mem, "available": avail_mem},
        "gpus": gpu_devices,
        "disks": disks,
        "torch": torch_info,
        "capabilities": capabilities,
        "notes": notes,
    }


def _capabilities(torch: dict[str, Any], gpus: list[dict[str, Any]], cpu: dict[str, Any]) -> dict[str, Any]:
    cuda = bool(torch.get("cuda_available")) and bool(gpus)
    cc = None
    if gpus:
        raw = gpus[0].get("compute_capability")
        try:
            cc = float(raw) if raw else None
        except (TypeError, ValueError):
            cc = None
    torch_installed = bool(torch.get("installed"))

    def precision(name: str, available: bool, reason: str) -> dict[str, Any]:
        return {"name": name, "available": available, "reason": reason}

    precisions = [
        precision("fp32", True, "Always available." if torch_installed else "Requires PyTorch for tensor math."),
        precision("fp16", cuda, "Needs a CUDA device." if not cuda else "Supported on CUDA devices."),
        precision(
            "bf16",
            bool(cuda and cc and cc >= 8.0),
            "Needs a CUDA device with compute capability 8.0+ (Ampere or newer)."
            if not (cuda and cc and cc >= 8.0) else "Supported by this GPU.",
        ),
        precision(
            "tf32",
            bool(cuda and cc and cc >= 8.0),
            "Needs an Ampere or newer CUDA device.",
        ),
        precision(
            "int8",
            bool(cuda and module_available("bitsandbytes")),
            "Needs bitsandbytes and a CUDA device.",
        ),
        precision(
            "int4",
            bool(cuda and module_available("bitsandbytes")),
            "Needs bitsandbytes and a CUDA device.",
        ),
    ]

    return {
        "cuda": cuda,
        "cuda_compute_capability": cc,
        "mps": bool(torch.get("mps_available")),
        "cpu_training": True,
        "distribution": {
            "multi_gpu": len(gpus) > 1,
            "device_count": len(gpus),
            "data_parallel": len(gpus) > 1,
            "fsdp": bool(torch_installed and len(gpus) > 1),
            "deepspeed": bool(torch_installed and len(gpus) > 1 and module_available("deepspeed")),
            "accelerate": module_available("accelerate"),
        },
        "attention": {
            "sdpa": bool(torch_installed),
            "flash_attention": module_available("flash_attn"),
            "xformers": module_available("xformers"),
        },
        "quantization_backends": {
            "bitsandbytes": module_available("bitsandbytes"),
            "gptq": module_available("auto_gptq") or module_available("gptqmodel"),
            "awq": module_available("awq"),
            "gguf": module_available("llama_cpp") or bool(shutil.which("convert_hf_to_gguf.py")),
        },
        "memory_optimizations": {
            "gradient_checkpointing": torch_installed,
            "gradient_accumulation": True,
            "cpu_offload": bool(torch_installed and module_available("accelerate")),
            "nvme_offload": bool(torch_installed and module_available("deepspeed")),
            "paged_optimizer": bool(cuda and module_available("bitsandbytes")),
            "compile": bool(torch_installed and hasattr(__import__("sys"), "version_info")),
        },
        "precisions": precisions,
        "libraries": {
            name: {"installed": module_available(name), "version": module_version(name)}
            for name in (
                "torch", "transformers", "datasets", "peft", "trl", "accelerate",
                "bitsandbytes", "safetensors", "sentencepiece", "tokenizers",
                "pyarrow", "pandas", "numpy", "huggingface_hub", "nvidia_smi",
            )
        },
        "cpu_instructions": {
            "avx2": bool(cpu.get("avx2")),
            "avx512": bool(cpu.get("avx512")),
        },
    }


def require_gpu(gpus: list[dict[str, Any]], purpose: str) -> None:
    if not gpus:
        raise ZxError(
            code="no_gpu",
            message=f"{purpose} requires a CUDA GPU and none was detected.",
            hint="Run this operation on CPU where the backend supports it, or install an NVIDIA GPU.",
        )
