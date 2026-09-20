"""Real hardware and CUDA detection.

Two independent sources are used on purpose:

* ``nvidia-smi`` — present whenever an NVIDIA driver works, even without
  PyTorch. Ground truth for "is there a usable GPU here".
* ``torch.cuda`` — ground truth for "can PyTorch actually use it", including
  the CUDA version the installed wheel was built against.

Nothing is synthesized: if a value cannot be read it stays ``None`` and the UI
shows an honest "unknown".
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

_SMI_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
    "driver_version",
    "compute_cap",
]


def _run(command: list[str], timeout: int = 12) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"not found: {command[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out: {' '.join(command)}"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)


def nvidia_smi_path() -> str | None:
    """Locate nvidia-smi, including the common Windows install paths."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for path in (
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
    ):
        if os.path.isfile(path):
            return path
    return None


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in ("n/a", "[n/a]", "unknown"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def gpus_from_smi() -> dict[str, Any]:
    """Query every NVIDIA GPU via nvidia-smi."""
    binary = nvidia_smi_path()
    if not binary:
        return {
            "available": False,
            "reason": "nvidia-smi was not found. Either there is no NVIDIA GPU "
                      "or the NVIDIA driver is not installed.",
            "gpus": [],
        }

    code, out, err = _run([
        binary, f"--query-gpu={','.join(_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return {
            "available": False,
            "reason": (err or out or f"nvidia-smi exited with code {code}").strip(),
            "gpus": [],
        }

    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_SMI_FIELDS):
            continue
        gpus.append({
            "index": int(_to_float(parts[0]) or 0),
            "name": parts[1],
            "memory_total_mb": _to_float(parts[2]),
            "memory_used_mb": _to_float(parts[3]),
            "memory_free_mb": _to_float(parts[4]),
            "utilization_gpu": _to_float(parts[5]),
            "temperature_c": _to_float(parts[6]),
            "driver_version": parts[7] or None,
            "compute_capability": parts[8] or None,
        })

    if not gpus:
        return {"available": False, "reason": "nvidia-smi returned no devices.", "gpus": []}

    return {"available": True, "binary": binary, "gpus": gpus}


def cuda_from_torch() -> dict[str, Any]:
    """What PyTorch itself reports. Imports torch lazily and defensively."""
    try:
        import torch  # noqa: PLC0415 - deliberate lazy import
    except Exception as exc:
        return {
            "torch_installed": False,
            "reason": f"PyTorch is not importable: {exc}",
            "available": False,
            "devices": [],
        }

    info: dict[str, Any] = {
        "torch_installed": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_build_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": None,
    }

    try:
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None and cudnn.is_available():
            info["cudnn_version"] = cudnn.version()
    except Exception:
        pass

    try:
        info["available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        info["available"] = False
        info["device_count"] = 0
        info["reason"] = str(exc)

    if info.get("available"):
        info["bf16_supported"] = bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    else:
        info["bf16_supported"] = False
        if not info.get("reason"):
            if not info["cuda_build_version"]:
                info["reason"] = (
                    "This PyTorch build has no CUDA support (CPU-only wheel). "
                    "Reinstall torch from the CUDA wheel index."
                )
            else:
                info["reason"] = (
                    "PyTorch has CUDA support but no usable device was found. "
                    "Check that the NVIDIA driver is installed and matches CUDA "
                    f"{info['cuda_build_version']}."
                )

    devices = []
    for index in range(int(info.get("device_count") or 0)):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024 ** 2), 1),
                "major": props.major,
                "minor": props.minor,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
        except Exception as exc:
            devices.append({"index": index, "error": str(exc)})
    info["devices"] = devices

    try:
        info["device_name_current"] = torch.cuda.get_device_name(0) if info.get("available") else None
    except Exception:
        info["device_name_current"] = None

    return info


def cpu_info() -> dict[str, Any]:
    physical = None
    try:
        import multiprocessing
        physical = multiprocessing.cpu_count()
    except Exception:
        pass

    model: str | None = None
    system = platform.system()
    if system == "Windows":
        code, out, _ = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ], timeout=15)
        if code == 0 and out.strip():
            model = out.strip().splitlines()[0].strip()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif system == "Darwin":
        code, out, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=8)
        if code == 0:
            model = out.strip()

    return {
        "model": model or platform.processor() or "Unknown CPU",
        "logical_cores": os.cpu_count(),
        "logical_cores_reported": physical,
        "architecture": platform.machine(),
    }


def memory_info() -> dict[str, Any]:
    total_mb = None
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in dir(os):
            page = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            total_mb = round(page * pages / (1024 ** 2), 1)
        elif platform.system() == "Windows":
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
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total_mb = round(status.ullTotalPhys / (1024 ** 2), 1)
    except Exception:
        total_mb = None

    return {"total_mb": total_mb}


def cuda_toolkit_version() -> str | None:
    """Version of a locally installed CUDA toolkit, if any (via nvcc)."""
    binary = shutil.which("nvcc")
    if not binary:
        return None
    code, out, _ = _run([binary, "--version"], timeout=10)
    if code != 0:
        return None
    match = re.search(r"release\s+([\d.]+)", out)
    return match.group(1) if match else None


def detect() -> dict[str, Any]:
    """Full hardware snapshot. Every field is observed, never assumed."""
    smi = gpus_from_smi()
    torch_info = cuda_from_torch()

    cuda_ready = bool(torch_info.get("available"))
    reasons: list[str] = []
    if not smi["available"]:
        reasons.append(smi.get("reason", "No NVIDIA GPU detected."))
    if not torch_info.get("torch_installed"):
        reasons.append("PyTorch is not installed, so GPU acceleration is unavailable yet.")
    elif not cuda_ready:
        reasons.append(torch_info.get("reason", "PyTorch cannot use a CUDA device."))

    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpu": smi,
        "cuda": torch_info,
        "cuda_toolkit": cuda_toolkit_version(),
        "nvidia_ready": smi["available"],
        "cuda_ready": cuda_ready,
        "training_device": "cuda" if cuda_ready else "cpu",
        "cuda_blockers": reasons,
    }
