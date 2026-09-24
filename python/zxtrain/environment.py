"""Python environment management.

The app never installs anything by itself: it discovers the interpreters that
exist, explains which one can actually run PyTorch, and performs an install only
when the user asks for it — into its own isolated environment, never into the
system Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .errors import ZxError
from .hardware import module_available, module_version
from .util import ensure_dir, human_bytes, now_iso, read_json, write_json_atomic

CORE_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "safetensors",
    "sentencepiece",
    "tokenizers",
    "numpy",
    "pyarrow",
    "huggingface_hub",
]
OPTIONAL_PACKAGES = ["trl", "bitsandbytes", "protobuf", "pillow", "soundfile", "librosa", "openpyxl", "PyYAML"]
# PyTorch publishes wheels for CPython 3.9 – 3.13. 3.14 (and any future major)
# is rejected up front instead of failing halfway through a 2 GB download.
TORCH_MAX_MINOR = 13


def interpreter_candidates() -> list[dict[str, Any]]:
    """Every Python this machine has, with a PyTorch compatibility verdict."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def probe(executable: str, label: str) -> None:
        try:
            resolved = str(Path(executable).resolve())
        except OSError:
            return
        if resolved in seen or not Path(resolved).exists():
            return
        seen.add(resolved)
        info = _probe_interpreter(resolved)
        if info:
            info["label"] = label
            found.append(info)

    probe(sys.executable, "engine interpreter")
    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3.9",
                 "python3", "python"):
        located = shutil.which(name)
        if located:
            probe(located, name)
    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            output = subprocess.run(
                [py_launcher, "-0p"], capture_output=True, text=True, timeout=10,
                creationflags=0x08000000 if os.name == "nt" else 0,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            output = ""
        for line in output.splitlines():
            if "-V:" in line:
                version = line.split("-V:")[1].split()[0]
                path = line.split(" ", 1)[1].strip() if " " in line else ""
                if path:
                    probe(path, f"python {version}")
    for candidate in (
        Path("C:/Python313/python.exe"), Path("C:/Python312/python.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python312/python.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python/Python313/python.exe",
        Path("/usr/bin/python3"), Path("/usr/local/bin/python3"),
    ):
        if candidate.exists():
            probe(str(candidate), candidate.name)
    return found


def _probe_interpreter(executable: str) -> dict[str, Any] | None:
    script = (
        "import json,sys;"
        "print(json.dumps({'version': '.'.join(map(str,sys.version_info[:3])),"
        "'major': sys.version_info[0],'minor': sys.version_info[1],"
        "'executable': sys.executable,'prefix': sys.prefix,"
        "'venv': sys.prefix != getattr(sys,'base_prefix',sys.prefix),"
        "'torch': __import__('importlib.util',fromlist=['x']).find_spec('torch') is not None}))"
    )
    try:
        output = subprocess.run(
            [executable, "-c", script], capture_output=True, text=True, timeout=25,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode != 0 or not output.stdout.strip():
        return None
    import json

    try:
        payload = json.loads(output.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    minor = int(payload.get("minor") or 0)
    payload["torch_compatible"] = minor <= TORCH_MAX_MINOR
    payload["torch_note"] = (
        "PyTorch has no wheels for this Python version yet — pick another interpreter."
        if not payload["torch_compatible"] else ""
    )
    return payload


def venv_path(workspace: Path) -> Path:
    return Path(workspace) / "runtime" / ".venv"


def venv_python(workspace: Path) -> Path:
    venv = venv_path(workspace)
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def report(workspace: str | Path) -> dict[str, Any]:
    workspace = Path(workspace)
    venv = venv_python(workspace)
    packages: dict[str, Any] = {}
    for name in CORE_PACKAGES + OPTIONAL_PACKAGES:
        packages[name] = {"installed": module_available(name), "version": module_version(name)}
    recommendations: list[str] = []
    python_minor = sys.version_info.minor
    torch_ready = bool(packages["torch"]["installed"])
    if not torch_ready:
        if python_minor > TORCH_MAX_MINOR:
            recommendations.append(
                f"The engine is running on Python {sys.version_info.major}.{python_minor}, which PyTorch "
                f"does not support yet. Install the ML runtime with a Python 3.12 or 3.13 interpreter."
            )
        else:
            recommendations.append(
                "PyTorch is not installed. Install the ML runtime to enable real Transformers training."
            )
    missing_optional = [name for name in ("trl", "bitsandbytes") if not packages[name]["installed"]]
    if missing_optional and torch_ready:
        recommendations.append(
            f"Optional: {', '.join(missing_optional)} — needed for preference training and 4/8-bit training."
        )
    report_payload = {
        "engine_python": {
            "executable": sys.executable,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "prefix": sys.prefix,
            "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        },
        "interpreters": interpreter_candidates(),
        "workspace_venv": {
            "path": str(venv_path(workspace)),
            "python": str(venv),
            "exists": venv.exists(),
        },
        "packages": packages,
        "ready": {
            "tiny_training": True,
            "transformers_training": bool(packages["torch"]["installed"] and packages["transformers"]["installed"]),
            "peft": bool(packages["peft"]["installed"]),
            "quantisation": bool(packages["bitsandbytes"]["installed"]),
            "preference_training": bool(packages["trl"]["installed"]),
            "parquet": bool(packages["pyarrow"]["installed"]),
            "excel": bool(packages["openpyxl"]["installed"]),
        },
        "recommendations": recommendations,
        "install_plan": install_plan(workspace),
        "reported_at": now_iso(),
    }
    return report_payload


def install_plan(workspace: str | Path, cuda: bool = False, extra: list[str] | None = None) -> dict[str, Any]:
    """The exact command an install would run — shown before anything is downloaded."""
    workspace = Path(workspace)
    python = venv_python(workspace)
    packages = list(CORE_PACKAGES)
    if extra:
        packages.extend(extra)
    index = "https://download.pytorch.org/whl/cu124" if cuda else "https://download.pytorch.org/whl/cpu"
    torch_line = " ".join(
        package if package != "torch" else f"torch --index-url {index}"
        for package in packages
    )
    return {
        "venv": str(venv_path(workspace)),
        "steps": [
            f'"{sys.executable}" -m venv "{venv_path(workspace)}"',
            f'"{python}" -m pip install --upgrade pip',
            f'"{python}" -m pip install {torch_line}',
        ],
        "packages": packages,
        "index_url": index,
        "notes": [
            "Installing downloads several hundred megabytes; a CUDA build is much larger.",
            "The install goes into the app's own environment and never touches the system Python.",
        ],
    }


def create_venv(workspace: str | Path, base_python: str | None, emit: Callable[[dict[str, Any]], None]) -> Path:
    workspace = Path(workspace)
    venv = venv_path(workspace)
    if venv.exists():
        emit({"type": "log", "level": "info", "message": f"Reusing existing environment at {venv}"})
        return venv
    ensure_dir(venv.parent)
    executable = base_python or sys.executable
    emit({"type": "log", "level": "info", "message": f"Creating virtual environment with {executable}"})
    _stream([executable, "-m", "venv", str(venv)], emit)
    write_json_atomic(venv.parent / "runtime.json", {
        "created_at": now_iso(), "base_python": executable, "python": str(venv_python(workspace)),
    })
    return venv


def install_runtime(workspace: str | Path, cuda: bool, base_python: str | None,
                    extra: list[str] | None, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    workspace = Path(workspace)
    create_venv(workspace, base_python, emit)
    python = venv_python(workspace)
    if not python.exists():
        raise ZxError(
            code="venv_failed",
            message="The isolated environment could not be created.",
            hint="Check that the selected interpreter can create virtual environments "
                 "(python -m venv works) and that the workspace is writable.",
        )
    emit({"type": "log", "level": "info", "message": "Upgrading pip in the isolated environment."})
    _stream([str(python), "-m", "pip", "install", "--upgrade", "pip"], emit)
    plan = install_plan(workspace, cuda, extra)
    packages = plan["packages"]
    emit({"type": "log", "level": "info",
          "message": f"Installing {len(packages)} packages from {plan['index_url']}"})
    torch_command = [str(python), "-m", "pip", "install", "torch", "--index-url", plan["index_url"]]
    _stream(torch_command, emit)
    others = [package for package in packages if package != "torch"]
    _stream([str(python), "-m", "pip", "install", *others], emit)
    emit({"type": "log", "level": "info", "message": "Verifying the installation."})
    verify = _capture([str(python), "-c",
                       "import torch, transformers, peft, datasets;"
                       "print(torch.__version__, transformers.__version__, peft.__version__);"
                       "print('cuda', torch.cuda.is_available())"])
    emit({"type": "log", "level": "info", "message": f"Installed: {verify.strip() or 'verification produced no output'}"})
    return {"venv": str(venv_path(workspace)), "python": str(python), "verify": verify.strip(),
            "finished_at": now_iso()}


def _stream(command: list[str], emit: Callable[[dict[str, Any]], None]) -> int:
    emit({"type": "log", "level": "info", "message": "$ " + " ".join(command)})
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except OSError as exc:
        raise ZxError(
            code="install_failed",
            message=f"Could not start the install: {exc}",
            hint="Check that the interpreter path still exists and that you are online.",
        ) from exc
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            emit({"type": "log", "level": "info", "message": line})
    code = process.wait()
    if code != 0:
        raise ZxError(
            code="install_failed",
            message=f"Installer exited with code {code}.",
            hint="Read the log above for the failing package. A common cause is a Python version "
                 "without available wheels.",
        )
    return code


def _capture(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=300,
                              creationflags=0x08000000 if os.name == "nt" else 0).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def runtime_info(workspace: str | Path) -> dict[str, Any]:
    return read_json(venv_path(Path(workspace)).parent / "runtime.json", {}) or {}


def disk_usage(workspace: str | Path) -> dict[str, Any]:
    workspace = Path(workspace)
    venv = venv_path(workspace)
    total = 0
    if venv.exists():
        for path in venv.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
    return {"venv_bytes": total, "venv_human": human_bytes(total), "venv": str(venv)}
