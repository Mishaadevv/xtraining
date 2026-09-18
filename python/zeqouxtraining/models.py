"""Model resolution and inspection.

Works for both a local model folder and a Hugging Face repository id. Parameter
counts are read from real files where possible:

* ``model.safetensors.index.json`` → exact tensor count and file list
* ``*.safetensors`` headers → exact parameter count, without loading weights
* ``config.json`` → architecture, context length, dtype

Nothing is invented: unknown values stay ``None`` and are surfaced as such.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "FP8": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

_HF_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


class ModelError(Exception):
    def __init__(self, message: str, *, hint: str = "", code: str = "model_error"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


def looks_like_hf_id(source: str) -> bool:
    """'meta-llama/Llama-3.2-1B' style identifiers (not an existing local path)."""
    if not source:
        return False
    if Path(source).exists():
        return False
    if re.match(r"^[A-Za-z]:[\\/]", source) or source.startswith(("/", "\\\\", ".")):
        return False
    return bool(_HF_ID_RE.match(source.strip()))


def read_config_json(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def safetensors_header_params(file_path: Path) -> int | None:
    """Exact parameter count from a .safetensors header.

    The format is: 8-byte little-endian header length, then a JSON header with
    every tensor's dtype and shape. Reading it costs a few kilobytes regardless
    of how large the model is.
    """
    try:
        with open(file_path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None
            length = struct.unpack("<Q", raw_length)[0]
            if length <= 0 or length > 200 * 1024 * 1024:
                return None
            header = json.loads(handle.read(length).decode("utf-8", errors="replace"))
    except Exception:
        return None

    total = 0
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        count = 1
        for dim in shape:
            count *= int(dim)
        total += count
    return total or None


def inspect(source: str, hf_cache_dir: str | None = None) -> dict[str, Any]:
    """Inspect a local folder or a Hugging Face id."""
    source = (source or "").strip()
    if not source:
        raise ModelError("No model source was given.", code="empty_source")

    if looks_like_hf_id(source):
        return _inspect_hf(source, hf_cache_dir)

    path = Path(source)
    if not path.exists():
        raise ModelError(
            f"'{source}' does not exist.",
            hint="Choose a model folder again, or use a Hugging Face id like 'meta-llama/Llama-3.2-1B'.",
            code="not_found",
        )
    if path.is_file():
        path = path.parent

    return _inspect_local(path)


def _inspect_local(path: Path) -> dict[str, Any]:
    config = read_config_json(path)
    weight_files = sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
    ) if path.is_dir() else []

    index_params = None
    index_path = path / "model.safetensors.index.json"
    shard_files: list[str] = []
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8", errors="replace"))
            index_params = int(index.get("metadata", {}).get("total_size") or 0) or None
            weight_map = index.get("weight_map") or {}
            shard_files = sorted(set(weight_map.values()))
        except Exception:
            index_params = None

    # Prefer exact per-file parameter counts from safetensors headers.
    params = None
    exact = False
    safetensors = [p for p in weight_files if p.suffix.lower() == ".safetensors"]
    if safetensors and not shard_files:
        total = 0
        got_any = False
        for file_path in safetensors:
            count = safetensors_header_params(file_path)
            if count:
                total += count
                got_any = True
        if got_any:
            params = total
            exact = True
    if params is None and shard_files:
        total = 0
        got_any = False
        for name in shard_files:
            file_path = path / name
            if not file_path.is_file():
                continue
            count = safetensors_header_params(file_path)
            if count:
                total += count
                got_any = True
        if got_any:
            params = total
            exact = True

    if params is None:
        params = estimate_params_from_config(config)
        exact = False

    size_bytes = sum(p.stat().st_size for p in weight_files) if weight_files else None
    quantized_detected = any(
        _weird_tensor_skip_line(key) for key in _peek_safetensors_keys(safetensors[:1])
    )

    fields = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size") or config.get("d_model") or config.get("n_embd"),
        "num_hidden_layers": (config.get("num_hidden_layers") or config.get("n_layer")
                              or config.get("num_layers")),
        "num_attention_heads": config.get("num_attention_heads") or config.get("n_head"),
        "vocab_size": config.get("vocab_size"),
        "intermediate_size": (config.get("intermediate_size")
                              or config.get("n_inner") or config.get("ffn_dim")),
        "torch_dtype": config.get("torch_dtype"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "max_position_embeddings": (config.get("max_position_embeddings")
                                    or config.get("n_positions")
                                    or config.get("max_seq_len")),
    }

    issues: list[dict[str, Any]] = []
    if not config:
        issues.append({
            "severity": "error",
            "code": "no_config",
            "message": "config.json is missing, so the architecture cannot be determined.",
            "hint": "Point at the folder that contains config.json and the model weights.",
        })
    if not weight_files:
        issues.append({
            "severity": "error",
            "code": "no_weights",
            "message": "No model weights were found in this folder.",
            "hint": "Expected one of: model.safetensors, *.bin, *.gguf.",
        })
    if not (path / "tokenizer_config.json").is_file() and not (path / "tokenizer.json").is_file():
        issues.append({
            "severity": "warning",
            "code": "no_tokenizer",
            "message": "No tokenizer files were found locally.",
            "hint": "Training will try to use the base model's tokenizer if one is identifiable.",
        })
    if any(p.suffix.lower() == ".gguf" for p in weight_files):
        issues.append({
            "severity": "warning",
            "code": "gguf_not_trainable",
            "message": "GGUF files are for inference, not for training with PyTorch.",
            "hint": "Training needs the original safetensors or .bin weights.",
        })

    return {
        "kind": "local",
        "source": str(path.resolve()),
        "name": path.name,
        "exists": True,
        "config": config,
        "fields": fields,
        "params": params,
        "params_exact": exact,
        "size_bytes": size_bytes,
        "weight_files": [p.name for p in weight_files],
        "shard_files": shard_files,
        "quantized_hint": quantized_detected,
        "trainable": bool(config and weight_files),
        "issues": issues,
        "max_position_embeddings": fields["max_position_embeddings"],
        # fp16/bf16 become available on CUDA regardless of the stored dtype.
        "training_dtypes": ["fp32", "fp16", "bf16"],
    }


def _weird_tensor_skip_line(key: str) -> bool:
    return key.endswith(".weight_scale") or key.endswith(".qweight")


def _peek_safetensors_keys(files: list[Path]) -> list[str]:
    keys: list[str] = []
    for file_path in files:
        try:
            with open(file_path, "rb") as handle:
                raw_length = handle.read(8)
                if len(raw_length) != 8:
                    continue
                length = struct.unpack("<Q", raw_length)[0]
                if length <= 0 or length > 200 * 1024 * 1024:
                    continue
                header = json.loads(handle.read(length).decode("utf-8", errors="replace"))
                keys.extend(k for k in header.keys() if k != "__metadata__")
        except Exception:
            continue
    return keys


def _inspect_hf(source: str, hf_cache_dir: str | None) -> dict[str, Any]:
    """Inspect a Hugging Face repo. Uses the local cache first, then the Hub."""
    issues: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    cached_path: str | None = None
    params = None
    exact = False

    try:
        from huggingface_hub import hf_hub_download, scan_cache_dir  # noqa: PLC0415
    except Exception:
        hf_hub_download = None  # type: ignore[assignment]
        scan_cache_dir = None  # type: ignore[assignment]
        issues.append({
            "severity": "warning",
            "code": "hub_missing",
            "message": "huggingface_hub is not installed, so the model can only be inspected at training time.",
            "hint": "Install the ML runtime in Settings → Environment to inspect and download models.",
        })

    if hf_hub_download is not None:
        try:
            config_file = hf_hub_download(
                repo_id=source, filename="config.json",
                cache_dir=hf_cache_dir, local_files_only=False,
            )
            config = json.loads(Path(config_file).read_text(encoding="utf-8", errors="replace"))
            cached_path = str(Path(config_file).parent)
            params = estimate_params_from_config(config)
        except Exception as exc:
            message = str(exc)
            if any(word in message.lower() for word in ("401", "403", "gated", "unauthorized")):
                issues.append({
                    "severity": "error",
                    "code": "gated_repo",
                    "message": f"'{source}' is gated and needs a Hugging Face access token.",
                    "hint": "Add a token in Settings → Hugging Face, then try again.",
                })
            elif any(word in message.lower() for word in ("connection", "offline", "timed out", "resolve")):
                issues.append({
                    "severity": "error",
                    "code": "offline",
                    "message": f"Could not reach Hugging Face for '{source}'.",
                    "hint": "Check your network connection, or use a local model folder.",
                })
            else:
                issues.append({
                    "severity": "error",
                    "code": "hf_error",
                    "message": f"Could not read '{source}' from Hugging Face: {message}",
                    "hint": "Check the repository id.",
                })

    fields = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "vocab_size": config.get("vocab_size"),
        "intermediate_size": config.get("intermediate_size"),
        "torch_dtype": config.get("torch_dtype"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "max_position_embeddings": config.get("max_position_embeddings"),
    }

    return {
        "kind": "huggingface",
        "source": source,
        "name": source.split("/")[-1],
        "exists": bool(config),
        "cached": bool(cached_path),
        "cached_path": cached_path,
        "config": config,
        "fields": fields,
        "params": params,
        "params_exact": exact,
        "size_bytes": None,
        "weight_files": [],
        "shard_files": [],
        "quantized_hint": False,
        "trainable": bool(config),
        "issues": issues,
        "max_position_embeddings": fields["max_position_embeddings"],
        "requires_download": not bool(cached_path),
    }


def estimate_params_from_config(config: dict[str, Any]) -> int | None:
    """Parameter estimate from the architecture config (approximate by design)."""
    if not config:
        return None
    hidden = config.get("hidden_size") or config.get("d_model") or config.get("n_embd")
    layers = (config.get("num_hidden_layers") or config.get("n_layer")
              or config.get("num_layers"))
    vocab = config.get("vocab_size")
    if not hidden or not layers:
        return None

    hidden = int(hidden)
    layers = int(layers)
    intermediate = config.get("intermediate_size") or config.get("n_inner") or hidden * 4
    intermediate = int(intermediate)

    attention = 4 * hidden * hidden
    # Llama/Mistral-style gated MLP (gate, up, down).
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    per_layer = attention + mlp + norms

    total = layers * per_layer
    if vocab:
        total += int(vocab) * hidden
        if not config.get("tie_word_embeddings", False):
            total += int(vocab) * hidden
    return int(total)


def default_output_name(source: str, method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (source or "model").split("/")[-1]).strip("-")
    return f"{slug or 'model'}-{method}"
