"""Model discovery, inspection and registry operations.

Inspection never loads weights into memory: configs, tokenizer files and weight
headers are read directly. Parameter counts therefore come from real tensor
headers (or are clearly labelled as estimates for legacy `.bin` checkpoints).
"""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any

from . import safetensors
from .errors import ZxError
from .util import (
    dir_size,
    ensure_dir,
    human_bytes,
    now_iso,
    read_json,
    safe_slug,
    tree_digest,
    write_json_atomic,
)

WEIGHT_PATTERNS = (
    "*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin", "*.bin",
    "model*.gguf", "*.gguf", "*.pt", "*.ckpt",
)
CONFIG_FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "adapter_config.json", "quantize_config.json", "chat_template.jinja")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "vocab.txt", "spiece.model")
ARCHITECTURE_NOTES = {
    "llama": "Llama-style decoder. Fully supported by the Transformers/PEFT backend.",
    "mistral": "Mistral-style decoder, sliding-window attention. Supported by the Transformers/PEFT backend.",
    "qwen2": "Qwen2 decoder. Supported by the Transformers/PEFT backend.",
    "qwen3": "Qwen3 decoder. Requires a recent Transformers release; check the Environment page.",
    "phi": "Phi decoder. Supported by the Transformers/PEFT backend.",
    "phi3": "Phi-3 decoder. Supported by the Transformers/PEFT backend.",
    "gemma": "Gemma decoder. Supported by the Transformers/PEFT backend.",
    "gemma2": "Gemma-2 decoder. Supported with recent Transformers; bf16 recommended.",
    "gpt2": "GPT-2 style decoder. Full support, including the pure-Python tiny backend for small sizes.",
    "gpt_neox": "GPT-NeoX decoder. Supported by the Transformers/PEFT backend.",
    "gptj": "GPT-J decoder. Supported by the Transformers/PEFT backend.",
    "falcon": "Falcon decoder. Higher memory use; 4-bit loading recommended on small GPUs.",
    "bloom": "BLOOM decoder. Supported by the Transformers/PEFT backend.",
    "opt": "OPT decoder. Supported by the Transformers/PEFT backend.",
    "mixtral": "Mixture-of-experts decoder. Trainable only with expert-aware setups; expect very high memory.",
    "t5": "Encoder-decoder. Training works through the Transformers backend, not the tiny backend.",
    "bert": "Encoder-only. Use classification fine-tuning; generation is not supported.",
    "whisper": "Speech encoder-decoder. Multimodal path; requires the audio extras.",
    "llava": "Vision-language model. Multimodal path; requires the vision extras.",
    "zx-tiny": "Model produced by the ZeqouXTraining tiny backend (pure Python). Fully supported for continuation training, evaluation and generation.",
}


def _read_text(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def read_gguf_metadata(path: Path, max_kv: int = 512) -> dict[str, Any]:
    """Parse a GGUF header for architecture/quantisation metadata."""
    import io

    scalars = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
               5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return {"error": "not a GGUF file"}
            version = struct.unpack("<I", handle.read(4))[0]
            tensor_count = struct.unpack("<Q", handle.read(8))[0]
            kv_count = struct.unpack("<Q", handle.read(8))[0]

            def read_string() -> str:
                length = struct.unpack("<Q", handle.read(8))[0]
                return handle.read(length).decode("utf-8", errors="replace")

            def read_value(kind: int) -> Any:
                if kind == 8:
                    return read_string()
                if kind == 9:
                    element_kind = struct.unpack("<I", handle.read(4))[0]
                    count = struct.unpack("<Q", handle.read(8))[0]
                    return [read_value(element_kind) for _ in range(min(count, 64))]
                spec = scalars.get(kind)
                if spec is None:
                    raise ValueError(f"unknown gguf value type {kind}")
                fmt, size = spec
                return struct.unpack(fmt, handle.read(size))[0]

            metadata: dict[str, Any] = {}
            for _ in range(min(kv_count, max_kv)):
                key = read_string()
                kind = struct.unpack("<I", handle.read(4))[0]
                try:
                    metadata[key] = read_value(kind)
                except (ValueError, struct.error, io.UnsupportedOperation):
                    break
        return {"version": version, "tensor_count": tensor_count, "kv_count": kv_count, "metadata": metadata}
    except (OSError, struct.error) as exc:
        return {"error": f"GGUF header could not be read: {exc}"}


def _bin_weight_bytes(path: Path) -> tuple[int, int]:
    """(uncompressed bytes, file count) inside a legacy .bin / .pt archive."""
    try:
        with zipfile.ZipFile(path) as archive:
            total = 0
            count = 0
            for info in archive.infolist():
                if info.filename.startswith("archive/data/"):
                    total += info.file_size
                    count += 1
            if count:
                return total, count
    except (zipfile.BadZipFile, OSError):
        pass
    try:
        return path.stat().st_size, 1
    except OSError:
        return 0, 0


def list_weight_files(root: Path) -> dict[str, Any]:
    root = Path(root)
    safetensors_files = sorted(root.glob("*.safetensors"))
    bin_files = [p for p in sorted(root.glob("*.bin")) if p.name != "training_args.bin"]
    gguf_files = sorted(root.glob("*.gguf")) + sorted(root.glob("*.GGUF"))
    other = [p for p in sorted(root.glob("*.pt")) + sorted(root.glob("*.ckpt"))]
    return {
        "safetensors": safetensors_files,
        "bin": bin_files,
        "gguf": gguf_files,
        "other": other,
    }


def detect_format(root: Path) -> dict[str, Any]:
    root = Path(root)
    files = list_weight_files(root)
    has_adapter = (root / "adapter_config.json").exists()
    if files["gguf"]:
        kind = "gguf"
    elif has_adapter and (files["safetensors"] or files["bin"]):
        kind = "peft-adapter"
    elif files["safetensors"]:
        kind = "transformers-safetensors"
    elif files["bin"]:
        kind = "transformers-pytorch-bin"
    elif files["other"]:
        kind = "raw-checkpoint"
    elif (root / "config.json").exists():
        kind = "config-only"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "weight_files": {key: [str(p) for p in value] for key, value in files.items()},
        "is_adapter": has_adapter,
    }


def _tokenizer_report(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"files": [], "class": None, "vocab_size": None, "special_tokens": {}}
    for name in TOKENIZER_FILES:
        path = root / name
        if path.exists():
            report["files"].append({"name": name, "size": path.stat().st_size})
    config = read_json(root / "tokenizer_config.json", {}) or {}
    report["class"] = config.get("tokenizer_class")
    report["model_max_length"] = config.get("model_max_length")
    report["chat_template"] = bool(config.get("chat_template"))
    report["added_tokens_decoder"] = len(config.get("added_tokens_decoder") or {}) or None
    specials = read_json(root / "special_tokens_map.json", {}) or {}
    report["special_tokens"] = {
        key: (value.get("content") if isinstance(value, dict) else value)
        for key, value in specials.items()
    }
    vocab_path = root / "vocab.json"
    if vocab_path.exists():
        vocab = read_json(vocab_path, {}) or {}
        report["vocab_size"] = len(vocab) or None
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            with tokenizer_json.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            vocab = (payload.get("model") or {}).get("vocab")
            if isinstance(vocab, dict) and vocab:
                report["vocab_size"] = len(vocab)
            report["tokenizer_type"] = (payload.get("model") or {}).get("type")
            report["merges"] = len((payload.get("model") or {}).get("merges") or []) or None
        except (OSError, json.JSONDecodeError):
            report["notes"] = "tokenizer.json exists but could not be parsed"
    return report


def _architecture_summary(config: dict[str, Any]) -> dict[str, Any]:
    archs = config.get("architectures") or ([config["model_type"]] if config.get("model_type") else [])
    model_type = str(config.get("model_type") or (archs[0] if archs else "unknown")).lower()
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads") or heads
    hidden = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    head_dim = config.get("head_dim") or (round(hidden / heads) if (hidden and heads) else None)
    return {
        "architectures": archs,
        "model_type": model_type,
        "hidden_size": hidden,
        "intermediate_size": config.get("intermediate_size") or config.get("ffn_dim"),
        "num_layers": config.get("num_hidden_layers") or config.get("n_layer"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings") or config.get("n_positions") or config.get("seq_length"),
        "rope_theta": (config.get("rope_theta") or (config.get("rope_scaling") or {}).get("rope_theta")),
        "rope_scaling": config.get("rope_scaling"),
        "hidden_act": config.get("hidden_act") or config.get("activation_function"),
        "norm": config.get("rms_norm_eps") and "rmsnorm" or (config.get("layer_norm_eps") and "layernorm" or None),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "torch_dtype": config.get("torch_dtype") or config.get("dtype"),
        "sliding_window": config.get("sliding_window"),
        "expert_config": {
            "num_experts": config.get("num_local_experts") or config.get("num_experts"),
            "experts_per_token": config.get("num_experts_per_tok"),
        } if (config.get("num_local_experts") or config.get("num_experts")) else None,
        "note": ARCHITECTURE_NOTES.get(model_type, "No engine note for this architecture yet — the compatibility "
                                              "check runs the real detection instead of assuming support."),
    }


def inspect(path: str | Path, with_digest: bool = False) -> dict[str, Any]:
    root = Path(path).expanduser()
    if not root.exists():
        raise ZxError(
            code="model_missing",
            message=f"Model path does not exist: {root}",
            hint="Re-import the model, or check that the drive is still connected.",
        )
    if root.is_file():
        if root.suffix.lower() == ".gguf":
            return _inspect_gguf_file(root)
        raise ZxError(
            code="not_a_model_dir",
            message=f"{root.name} is a file, not a model directory.",
            hint="Import the folder that contains config.json, or convert this file first.",
        )

    fmt = detect_format(root)
    config = read_json(root / "config.json", {}) or {}
    generation = read_json(root / "generation_config.json", {}) or {}
    adapter = read_json(root / "adapter_config.json", {}) or {}

    weight_summary: dict[str, Any] = {
        "parameter_count": None,
        "weight_bytes": 0,
        "dtypes": {},
        "tensor_count": 0,
        "source": None,
        "errors": [],
    }
    if fmt["weight_files"]["safetensors"]:
        summary = safetensors.summarise([Path(p) for p in fmt["weight_files"]["safetensors"]])
        weight_summary.update({
            "parameter_count": summary["parameter_count"],
            "weight_bytes": summary["weight_bytes"],
            "dtypes": summary["dtypes"],
            "tensor_count": summary["tensor_count"],
            "metadata": summary.get("metadata", {}),
            "source": "safetensors header (exact)",
            "errors": summary["errors"],
        })
    elif fmt["weight_files"]["bin"]:
        total = 0
        files = 0
        for raw in fmt["weight_files"]["bin"]:
            size, count = _bin_weight_bytes(Path(raw))
            total += size
            files += count
        dtype = str(config.get("torch_dtype") or "float32").lower()
        bits = 16 if dtype in ("bfloat16", "float16", "half") else 32
        weight_summary.update({
            "parameter_count": int(total * 8 / bits),
            "weight_bytes": total,
            "source": f"estimated from {files} legacy .bin archives at {dtype} (approximate)",
            "dtypes": {dtype: int(total * 8 / bits)},
        })
    elif fmt["weight_files"]["gguf"]:
        gguf = read_gguf_metadata(Path(fmt["weight_files"]["gguf"][0]))
        meta = gguf.get("metadata", {})
        weight_summary.update({
            "parameter_count": _gguf_parameter_count(meta),
            "weight_bytes": sum(p.stat().st_size for p in map(Path, fmt["weight_files"]["gguf"])),
            "source": "GGUF header",
            "gguf": {k: v for k, v in gguf.items() if k != "metadata"},
            "gguf_metadata": _gguf_selected(meta),
        })

    size_bytes, file_count = dir_size(root)
    report = {
        "path": str(root),
        "name": root.name,
        "format": fmt,
        "size_bytes": size_bytes,
        "file_count": file_count,
        "architecture": _architecture_summary(config),
        "config": config,
        "generation_config": generation,
        "tokenizer": _tokenizer_report(root),
        "weights": weight_summary,
        "adapter": adapter or None,
        "adapter_details": _adapter_details(adapter) if adapter else None,
        "files": _file_listing(root),
        "estimated_vram": _estimated_vram(weight_summary, config),
        "inspected_at": now_iso(),
    }
    if with_digest:
        report["digest"] = tree_digest(root)
    return report


def _inspect_gguf_file(path: Path) -> dict[str, Any]:
    gguf = read_gguf_metadata(path)
    meta = gguf.get("metadata", {})
    size = path.stat().st_size
    return {
        "path": str(path),
        "name": path.stem,
        "format": {"kind": "gguf", "weight_files": {"gguf": [str(path)]}, "is_adapter": False},
        "size_bytes": size,
        "file_count": 1,
        "architecture": {
            "architectures": [str(meta.get("general.architecture", "gguf"))],
            "model_type": str(meta.get("general.architecture", "gguf")).lower(),
            "num_layers": meta.get("block_count"),
            "hidden_size": meta.get("embedding_length"),
            "num_attention_heads": meta.get("attention.head_count"),
            "num_key_value_heads": meta.get("attention.head_count_kv"),
            "vocab_size": len(meta.get("tokenizer.ggml.tokens") or []) or None,
            "max_position_embeddings": meta.get("context_length"),
            "note": "GGUF weights are inference-ready. Use the GGUF runtime for generation; "
                    "training requires the original Transformers checkpoint.",
        },
        "config": {"quantization": _gguf_selected(meta)},
        "generation_config": {},
        "tokenizer": {"files": [], "class": "gguf-embedded", "vocab_size": len(meta.get("tokenizer.ggml.tokens") or []) or None},
        "weights": {
            "parameter_count": _gguf_parameter_count(meta),
            "weight_bytes": size,
            "source": "GGUF header",
            "gguf": {k: v for k, v in gguf.items() if k != "metadata"},
        },
        "files": [{"name": path.name, "size": size}],
        "estimated_vram": None,
        "inspected_at": now_iso(),
    }


def _gguf_selected(meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "general.architecture", "general.name", "general.size_label", "general.file_type",
        "block_count", "context_length", "embedding_length", "attention.head_count",
        "attention.head_count_kv", "rope.freq_base", "attention.layer_norm_rms_epsilon",
        "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
        "quantization.version", "general.quantization_version",
    )
    return {key: meta[key] for key in keys if key in meta}


def _gguf_parameter_count(meta: dict[str, Any]) -> int | None:
    """GGUF stores tensor shapes in the tensor table; the file size is a safe floor."""
    try:
        file_type = int(meta.get("general.file_type", -1))
    except (TypeError, ValueError):
        file_type = -1
    bits = {
        0: 32, 1: 16, 2: 18, 3: 19, 7: 8.5, 8: 8.5, 9: 10.5, 10: 5.5,
        12: 8.5, 13: 4.5, 14: 4.5, 15: 6.5, 16: 6.5, 17: 8.5, 18: 4.6,
    }.get(file_type)
    if bits is None:
        return None
    size = meta.get("__file_bytes__")
    return int(size * 8 / bits) if isinstance(size, int) else None


def _adapter_details(adapter: dict[str, Any]) -> dict[str, Any]:
    return {
        "peft_type": adapter.get("peft_type"),
        "task_type": adapter.get("task_type"),
        "base_model": adapter.get("base_model_name_or_path"),
        "r": adapter.get("r"),
        "lora_alpha": adapter.get("lora_alpha"),
        "lora_dropout": adapter.get("lora_dropout"),
        "target_modules": adapter.get("target_modules"),
        "bias": adapter.get("bias"),
        "modules_to_save": adapter.get("modules_to_save"),
        "use_rslora": adapter.get("use_rslora"),
        "init_lora_weights": adapter.get("init_lora_weights"),
        "scaling": (
            (float(adapter.get("lora_alpha", 0)) / float(adapter.get("r", 1)))
            if adapter.get("lora_alpha") and adapter.get("r") else None
        ),
    }


def _file_listing(root: Path, limit: int = 300) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= limit:
            break
        if path.is_file():
            try:
                entries.append({
                    "name": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "size_human": human_bytes(path.stat().st_size),
                })
            except OSError:
                continue
    return entries


def _estimated_vram(weights: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    params = weights.get("parameter_count")
    if not params:
        return None
    dtype = str(config.get("torch_dtype") or "float32").lower()
    weight_bits = 32 if "float32" in dtype else (16 if dtype else 32)
    base = params * weight_bits / 8
    return {
        "labelled": "estimated",
        "weights_only": int(base),
        "fp16_inference": int(params * 2),
        "int8_inference": int(params * 1.15),
        "int4_inference": int(params * 0.6),
        "lora_training_fp16": int(params * 2 + params * 0.05),
        "full_training_fp16": int(params * 2 * 4.5),
        "formula": "weights + gradients + optimizer + activations, assume 2 bytes/param at fp16",
    }


def scan(roots: list[str], max_depth: int = 3) -> list[dict[str, Any]]:
    """Find model directories under the given roots without following symlink loops."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in roots:
        root = Path(raw).expanduser()
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for current, dirs, files in _walk(root, max_depth):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
            if (Path(current) / "config.json").exists() or any(f.endswith(".gguf") for f in files):
                resolved = str(Path(current).resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append({
                    "path": resolved,
                    "name": Path(current).name,
                    "depth": len(Path(current).parts) - base_depth,
                    "has_weights": any(
                        f.endswith((".safetensors", ".bin", ".gguf", ".pt")) for f in files
                    ),
                })
    return found


def _walk(root: Path, max_depth: int):
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        dirs = [entry.name for entry in entries if entry.is_dir()]
        files = [entry.name for entry in entries if entry.is_file()]
        yield str(current), dirs, files
        if depth + 1 >= max_depth:
            continue
        for name in dirs:
            stack.append((current / name, depth + 1))


def import_into(src: str | Path, destination_root: Path, copy: bool = True, name: str | None = None) -> Path:
    """Bring an external model folder into the workspace (copy by default)."""
    source = Path(src).expanduser()
    if not source.exists():
        raise ZxError(
            code="model_missing",
            message=f"Cannot import {source}: path does not exist.",
            hint="Pick the folder that contains config.json or the model file.",
        )
    target_root = ensure_dir(Path(destination_root))
    target = target_root / safe_slug(name or source.stem, "model")
    counter = 2
    while target.exists():
        target = target_root / f"{safe_slug(name or source.stem, 'model')}-{counter}"
        counter += 1
    try:
        if source.is_dir():
            if copy:
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.move(str(source), str(target))
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / source.name)
    except OSError as exc:
        raise ZxError(
            code="import_failed",
            message=f"Could not import {source.name}: {exc}",
            hint="Check free disk space and that the source file is not locked by another program.",
        ) from exc
    write_json_atomic(target / ".zxtrain-import.json", {"imported_from": str(source), "imported_at": now_iso()})
    return target
