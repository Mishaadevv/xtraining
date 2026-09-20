"""Dependency inspection for the ML runtime.

Uses ``importlib.metadata``/``find_spec`` only — never imports torch — so this
runs on a bare interpreter and can say exactly what is missing and how to fix it.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import platform
import sys
from typing import Any

# Grouped so the UI can explain *why* each package matters.
PACKAGES: list[dict[str, str]] = [
    {"module": "torch", "pip": "torch", "group": "core",
     "purpose": "Training and inference engine"},
    {"module": "transformers", "pip": "transformers", "group": "core",
     "purpose": "Model and tokenizer loading"},
    {"module": "peft", "pip": "peft", "group": "core",
     "purpose": "LoRA / QLoRA adapters"},
    {"module": "accelerate", "pip": "accelerate", "group": "core",
     "purpose": "Device placement and mixed precision"},
    {"module": "safetensors", "pip": "safetensors", "group": "core",
     "purpose": "Safe weight file format"},
    {"module": "numpy", "pip": "numpy", "group": "core",
     "purpose": "Numerical primitives"},
    {"module": "huggingface_hub", "pip": "huggingface_hub", "group": "download",
     "purpose": "Downloading models and datasets from Hugging Face"},
    {"module": "bitsandbytes", "pip": "bitsandbytes", "group": "quantization",
     "purpose": "4-bit / 8-bit quantization for QLoRA (NVIDIA CUDA)"},
    {"module": "datasets", "pip": "datasets", "group": "datasets",
     "purpose": "Hugging Face datasets and folder loading"},
    {"module": "pyarrow", "pip": "pyarrow", "group": "datasets",
     "purpose": "Reading .parquet dataset files"},
    {"module": "sentencepiece", "pip": "sentencepiece", "group": "tokenizer",
     "purpose": "Tokenizer for Llama/Mistral-style models"},
]

# Which packages gate which capability, so the UI answers "can I run this?"
# before a run starts instead of failing after loading weights.
CAPABILITIES: dict[str, list[str]] = {
    "lora": ["torch", "transformers", "peft", "accelerate", "datasets"],
    "qlora": ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"],
    "sft": ["torch", "transformers", "peft", "accelerate", "datasets"],
    "full": ["torch", "transformers", "accelerate", "datasets"],
    # From scratch needs no peft: the model is built in code.
    "scratch": ["torch", "transformers", "accelerate", "datasets"],
    "inference": ["torch", "transformers"],
    "hf_datasets": ["datasets"],
    "parquet": ["pyarrow"],
    "hf_download": ["huggingface_hub"],
}

INSTALL_GROUPS = ("core", "quantization", "download", "datasets", "tokenizer")


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(module: str) -> str | None:
    """Installed distribution version, if the module is importable."""
    for name in (module, module.replace("_", "-")):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
        except Exception:
            continue
    return "installed"


def inspect() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for entry in PACKAGES:
        module = entry["module"]
        installed = _available(module)
        packages[module] = {
            "installed": installed,
            "version": _version(module) if installed else None,
            "pip": entry["pip"],
            "group": entry["group"],
            "purpose": entry["purpose"],
        }

    missing_core = [p["pip"] for p in PACKAGES
                    if p["group"] == "core" and not packages[p["module"]]["installed"]]

    capabilities = {
        name: {
            "ready": all(packages.get(m, {}).get("installed") for m in required),
            "requires": required,
            "missing": [m for m in required if not packages.get(m, {}).get("installed")],
        }
        for name, required in CAPABILITIES.items()
    }

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "missing_core": missing_core,
        "capabilities": capabilities,
        "training_ready": not missing_core,
    }


def install_plan(cuda_tag: str | None = None) -> dict[str, Any]:
    """Build the exact pip command that would make this machine training-ready.

    For NVIDIA GPUs torch must come from the CUDA wheel index; a plain
    ``pip install torch`` may pull a CPU-only build.
    """
    missing = [p["pip"] for p in PACKAGES
               if p["group"] in INSTALL_GROUPS and not _available(p["module"])]

    args = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if cuda_tag:
        # Options BEFORE packages: torch comes from the CUDA index while
        # everything else must still resolve from PyPI.
        args += ["--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
                 "--extra-index-url", "https://pypi.org/simple"]
        args.append("torch")
        missing = [m for m in missing if m != "torch"]
    args += missing or ["torch", "transformers", "peft", "accelerate", "safetensors"]

    return {
        "command": " ".join(_quote(a) for a in args),
        "argv": args,
        "packages": missing,
        "cuda_tag": cuda_tag,
        "note": (
            "Run this with the interpreter shown above. For NVIDIA GPUs the "
            "CUDA wheel index installs a build with full GPU support; a plain "
            "'pip install torch' may pull a CPU-only build."
        ),
    }


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value or "\\" in value else value
