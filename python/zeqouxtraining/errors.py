"""Turn raw exceptions into explanations a user can act on.

The requirement is explicit: no wall of traceback in the main UI. The traceback
is still preserved and sent to the technical log panel, but what the user reads
first is a sentence in plain language and a suggested next step.
"""

from __future__ import annotations

import traceback
from typing import Any

# (regex-free) substring match -> (code, message template, hint)
_PATTERNS: list[tuple[tuple[str, ...], str, str, str]] = [
    (
        ("cuda out of memory", "cuda oom", "out of memory on device"),
        "cuda_oom",
        "The GPU ran out of memory while loading or training.",
        "Lower the batch size, switch to QLoRA (4-bit), reduce the context length, "
        "or enable gradient checkpointing. The Hardware screen shows what the current "
        "configuration is estimated to need.",
    ),
    (
        ("out of memory", "cannot allocate memory", "memoryerror"),
        "ram_oom",
        "The machine ran out of system memory.",
        "Lower the batch size and the context length, or use a smaller model.",
    ),
    (
        ("no kernel image is available", "not compatible with the current pytorch"),
        "cuda_arch_mismatch",
        "The installed PyTorch build does not support this GPU's compute capability.",
        "Reinstall torch from the CUDA wheel index in Settings → Environment.",
    ),
    (
        ("no cuda gpus are available", "torch not compiled with cuda", "cuda unknown error"),
        "cuda_unavailable",
        "PyTorch cannot use a CUDA device in this environment.",
        "The Hardware screen shows the exact reason. A CPU-only torch build is the "
        "most common cause; reinstall it from the CUDA wheel index.",
    ),
    (
        ("bitsandbytes",),
        "bitsandbytes_error",
        "The 4-bit/8-bit quantization backend (bitsandbytes) failed.",
        "QLoRA needs bitsandbytes and an NVIDIA GPU. If you do not have one, switch "
        "the method to LoRA.",
    ),
    (
        ("gated repo", "401 client error", "403 client error", "cannot access gated",
         "unauthorized", "requires you to be authenticated"),
        "gated_model",
        "This model is gated: Hugging Face requires an accepted licence and an access token.",
        "Accept the licence on the model page, then add your token in Settings → Hugging Face.",
    ),
    (
        ("connectionerror", "failed to establish", "max retries exceeded",
         "temporary failure in name resolution", "connection reset", "timed out",
         "proxyerror", "ssl"),
        "network",
        "A network request failed while downloading the model or dataset.",
        "Check your connection or proxy. Everything already in the local cache keeps working offline.",
    ),
    (
        ("does not appear to have a file named", "no such file or directory",
         "cannot find the requested files", "is not a local folder"),
        "missing_files",
        "A required model or tokenizer file could not be found or downloaded.",
        "Confirm the model id, or point at a local folder that contains config.json "
        "and the weight files.",
    ),
    (
        ("size mismatch", "error(s) in loading state_dict", "unexpected key"),
        "state_dict_mismatch",
        "The weight files do not match the architecture described by config.json.",
        "The model folder is likely incomplete or the files come from different versions.",
    ),
    (
        ("trust_remote_code", "remote code"),
        "remote_code",
        "This model needs custom code from its repository to load.",
        "Enable 'Allow custom model code' in Settings → Advanced, only for models you trust.",
    ),
    (
        ("is not a valid model identifier", "repo id must be", "invalid model identifier"),
        "bad_model_id",
        "The model identifier is not valid.",
        "Use the form 'owner/name', or select a local folder.",
    ),
    (
        ("tokenizer",),
        "tokenizer_error",
        "The tokenizer for this model could not be loaded.",
        "Some models need 'sentencepiece' or 'protobuf'. Install them in Settings → Environment.",
    ),
    (
        ("nan", "inf"),
        "diverged",
        "Training produced NaN values, which means the run diverged.",
        "Lower the learning rate (try 1e-4), increase warmup, and check the dataset for "
        "empty or malformed samples.",
    ),
]

_DEPENDENCY_HINT = (
    "Open Settings → Environment and install the ML runtime with the generated command."
)


def humanize(exc: BaseException, stage: str = "") -> dict[str, Any]:
    """Map an exception to {code, message, hint, traceback, stage}."""
    raw = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    lowered = raw.lower()

    # Domain errors (DatasetError, ExportError, …) are already written for the
    # user: keep their wording and hint instead of falling through to the
    # generic traceback message.
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code and hasattr(exc, "hint") \
            and not isinstance(exc, (ImportError, ModuleNotFoundError)):
        return {
            "code": code,
            "message": getattr(exc, "message", str(exc)) or str(exc),
            "hint": getattr(exc, "hint", "") or "",
            "traceback": traceback.format_exc(),
            "stage": stage,
        }

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        missing = getattr(exc, "name", None) or _first_missing(lowered)
        return {
            "code": "missing_dependency",
            "message": f"A required Python package is missing: {missing or 'unknown'}.",
            "hint": _DEPENDENCY_HINT,
            "traceback": traceback.format_exc(),
            "stage": stage,
        }

    for needles, code, message, hint in _PATTERNS:
        if any(needle in lowered for needle in needles):
            return {
                "code": code,
                "message": message,
                "hint": hint,
                "traceback": traceback.format_exc(),
                "stage": stage,
                "technical": raw,
            }

    if isinstance(exc, FileNotFoundError):
        return {
            "code": "file_not_found",
            "message": f"A required file was not found: {getattr(exc, 'filename', raw)}",
            "hint": "Check that the dataset and model paths still exist.",
            "traceback": traceback.format_exc(),
            "stage": stage,
        }

    if isinstance(exc, PermissionError):
        return {
            "code": "permission",
            "message": "A file or folder could not be written to.",
            "hint": "Check the storage folder in Settings → Storage and its permissions.",
            "traceback": traceback.format_exc(),
            "stage": stage,
        }

    return {
        "code": "unknown",
        "message": raw,
        "hint": "The full traceback is in the technical log. This is usually a model, "
                "dataset or dependency problem.",
        "traceback": traceback.format_exc(),
        "stage": stage,
    }


def _first_missing(lowered: str) -> str | None:
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                 "datasets", "safetensors", "sentencepiece", "pyarrow", "numpy"):
        if f"'{name}'" in lowered or f'"{name}"' in lowered or f" {name} " in lowered:
            return name
    return None
