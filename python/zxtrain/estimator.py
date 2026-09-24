"""Resource planning: how much VRAM, RAM, disk and time a run needs.

Every number produced here is an estimate and is labelled as such in the payload
(`"labelled": "estimated"`). Parameter counts taken from real weight headers are
passed through as exact values, and the estimator says which is which.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import datasets as ds
from .bpe import BPETokenizer
from .models import inspect as inspect_model
from .util import human_bytes

FULL_FT_METHODS = {"full_finetune", "continued_pretraining", "continued_training", "scratch"}
ADAPTER_METHODS = {"lora", "qlora", "adapter", "sft"}


def _config_params(config: dict[str, Any]) -> int | None:
    """Transformer parameter count from config.json alone (no weights needed)."""
    try:
        vocab = int(config.get("vocab_size") or 0)
        hidden = int(config.get("hidden_size") or config.get("n_embd") or config.get("d_model") or 0)
        layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
        intermediate = int(config.get("intermediate_size") or config.get("n_inner") or hidden * 4)
        heads = int(config.get("num_attention_heads") or 0) or 1
        kv_heads = int(config.get("num_key_value_heads") or heads)
        head_dim = int(config.get("head_dim") or (hidden // heads if heads else hidden))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not (vocab and hidden and layers):
        return None
    embeddings = vocab * hidden * (2 if config.get("tie_word_embeddings") else 1)
    attention = layers * (hidden * hidden + 2 * hidden * kv_heads * head_dim + hidden * hidden)
    mlp = layers * (3 * hidden * intermediate if config.get("hidden_act") in ("silu", "swiglu", "gelu_pytorch_tanh")
                    else 2 * hidden * intermediate)
    norms = layers * 2 * hidden
    return int(embeddings + attention + mlp + norms)


def _bytes_per_param(precision: str, quantization: str) -> float:
    if quantization == "int4":
        return 0.5
    if quantization == "int8":
        return 1.0
    return {"fp32": 4.0, "float32": 4.0, "fp16": 2.0, "float16": 2.0,
            "bf16": 2.0, "bfloat16": 2.0, "auto": 2.0}.get(str(precision).lower(), 4.0)


def _adapter_trainable(config: dict[str, Any], rank: int, target_modules: list[str] | None) -> int | None:
    hidden = config.get("hidden_size") or config.get("n_embd")
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    intermediate = config.get("intermediate_size") or (hidden * 4 if hidden else None)
    heads = config.get("num_attention_heads") or 1
    kv_heads = config.get("num_key_value_heads") or heads
    if not (hidden and layers):
        return None
    head_dim = config.get("head_dim") or (hidden // heads)
    module_sizes = {
        "q_proj": (hidden, heads * head_dim),
        "k_proj": (hidden, kv_heads * head_dim),
        "v_proj": (hidden, kv_heads * head_dim),
        "o_proj": (heads * head_dim, hidden),
        "out_proj": (heads * head_dim, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
        "fc1": (hidden, intermediate),
        "fc2": (intermediate, hidden),
        "dense": (hidden, hidden),
        "c_attn": (hidden, 3 * hidden),
        "c_proj": (hidden, hidden),
        "c_fc": (hidden, 4 * hidden),
        "query_key_value": (hidden, 3 * hidden),
        "dense_h_to_4h": (hidden, 4 * hidden),
        "dense_4h_to_h": (4 * hidden, hidden),
        "qkv_proj": (hidden, (heads + 2 * kv_heads) * head_dim),
        "gate_up_proj": (hidden, 2 * intermediate),
    }
    total = 0
    chosen = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name in chosen:
        size = module_sizes.get(name)
        if not size:
            continue
        total += (size[0] + size[1]) * rank * layers
    return total or None


def _dataset_tokens(paths: list[str], mapping: dict[str, Any], tokenizer_path: str | None,
                    sample_records: int = 2000) -> dict[str, Any]:
    """Token counts: exact when a tokenizer is available, estimated otherwise."""
    tokenizer = None
    if tokenizer_path and Path(tokenizer_path).exists():
        try:
            tokenizer = BPETokenizer.load(Path(tokenizer_path))
        except Exception:
            tokenizer = None
    total_records = 0
    sampled_chars = 0
    sample_counts: list[int] = []
    per_dataset: list[dict[str, Any]] = []
    for path in paths:
        texts = ds.extract_texts(Path(path), mapping, limit=sample_records)
        characters = sum(len(text) for text in texts)
        counts = [len(tokenizer.encode(text)) for text in texts] if tokenizer else []
        sampled_chars += characters
        if tokenizer:
            total_records += len(texts)
            sample_counts.extend(counts)
        else:
            full_report = None
            try:
                full_report = ds.inspect_dataset(path, mapping, sample_size=200)
            except Exception:
                full_report = None
            count = (full_report or {}).get("record_count") or len(texts)
            total_records += int(count)
            per_dataset.append({"path": path, "records": count,
                                "sampled_characters": characters})
    if tokenizer and sample_counts:
        average_tokens = sum(sample_counts) / len(sample_counts)
        tokens = int(average_tokens * total_records)
        exact = True
        basis = f"measured with the tokenizer on {len(sample_counts)} sampled records"
    else:
        average_tokens = sampled_chars / max(1, len(paths) * sample_records) / 4 if sampled_chars else 0
        tokens = int(sampled_chars / 4 * max(1, total_records / max(1, sample_records * len(paths)))) \
            if sampled_chars else 0
        exact = False
        basis = "characters / 4 heuristic (no tokenizer selected)"
    return {
        "records": total_records,
        "tokens": tokens,
        "average_tokens_per_record": round(average_tokens, 1),
        "exact": exact,
        "basis": basis,
        "per_dataset": per_dataset,
    }


def plan_run(request: dict[str, Any], hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce the pre-flight plan the UI shows before a run starts."""
    method = str(request.get("method") or "lora")
    precision = str(request.get("precision") or "fp32")
    quantization = str(request.get("quantization") or "none")
    sequence_length = int(request.get("sequence_length") or 256)
    batch_size = int(request.get("batch_size") or 1)
    accumulation = int(request.get("gradient_accumulation") or 1)
    epochs = float(request.get("epochs") or 1.0)
    rank = int(request.get("lora_rank") or 8)
    checkpoint_limit = int(request.get("checkpoint_limit") or 3)
    save_every = int(request.get("save_every") or 100)
    paths = [str(p) for p in request.get("dataset_paths") or []]
    mapping = request.get("mapping") or {}
    model_path = request.get("base_model") or request.get("parent_checkpoint")
    warnings: list[str] = []
    exact_params = None
    config: dict[str, Any] = {}
    weight_bytes = 0

    if model_path and Path(str(model_path)).exists():
        try:
            report = inspect_model(model_path)
            config = report.get("config") or {}
            exact_params = (report.get("weights") or {}).get("parameter_count")
            weight_bytes = int((report.get("weights") or {}).get("weight_bytes") or 0)
            if (report.get("adapter_config") or (report.get("adapter") or {})):
                config = config or {}
        except Exception as exc:
            warnings.append(f"Model inspection failed, estimates use configuration defaults: {exc}")
    if not exact_params:
        exact_params = _config_params(config) if config else int(
            request.get("vocab_size") or 512
        ) * int(request.get("hidden_size") or 32) * (int(request.get("context_length") or 8) + 2)

    tokens = _dataset_tokens(paths, mapping, request.get("tokenizer_path"))
    if tokens["tokens"] == 0:
        warnings.append("No dataset tokens measured yet — select a dataset before starting the run.")

    is_full = method in FULL_FT_METHODS
    is_adapter = method in ADAPTER_METHODS
    per_param = _bytes_per_param(precision, quantization)
    trainable = exact_params
    if is_adapter and method != "sft":
        adapter_params = _adapter_trainable(config, rank, request.get("target_modules"))
        if adapter_params:
            trainable = adapter_params
        else:
            trainable = max(1, int(exact_params * 0.01))
            warnings.append(
                "Adapter size could not be derived from the model config; 1% of parameters assumed."
            )

    weights = weight_bytes or exact_params * per_param
    if quantization == "int4":
        weights = exact_params * 0.5
    elif quantization == "int8":
        weights = exact_params * 1.0

    gradients = trainable * per_param if is_full else trainable * 2
    if is_adapter and method != "sft":
        optimizer = trainable * 8 + trainable * 4  # Adam m/v + master weights
    else:
        optimizer = exact_params * 12
    activations = (batch_size * sequence_length * max(1, int(config.get("hidden_size") or
                   config.get("n_embd") or request.get("hidden_size") or 256)) *
                   max(1, int(config.get("num_hidden_layers") or request.get("num_hidden_layers") or 12)) *
                   (2 if per_param <= 2 else 4) * 1.6)
    overhead = 0.15
    vram_total = int((weights + gradients + optimizer + activations) * (1 + overhead))

    gpus = hardware.get("gpus") if hardware else None
    gpu_memory = int(gpus[0]["memory_total"]) if gpus and gpus[0].get("memory_total") else None
    ram_total = int((hardware or {}).get("memory", {}).get("total") or 0) or None
    ram_available = int((hardware or {}).get("memory", {}).get("available") or 0) or None
    ram_needed = vram_total if not gpu_memory else int(weights + activations + optimizer * 0.2)

    effective_batch = batch_size * accumulation
    tokens_per_step = effective_batch * sequence_length
    steps = int((tokens["tokens"] / tokens_per_step)) if tokens_per_step and tokens["tokens"] else 0
    steps = max(1, int(steps * max(0.01, epochs))) if steps else 0
    checkpoints_planned = max(1, steps // max(1, save_every)) if steps else 0
    checkpoint_bytes = (int(weights) + int(optimizer)) if is_full else int(weights * 0.02 + trainable * 12)
    disk_needed = int(checkpoint_bytes * min(checkpoint_limit, max(1, checkpoints_planned)) * 1.1)

    risk = "unknown"
    notes: list[str] = []
    if gpu_memory:
        if vram_total > gpu_memory:
            risk = "will_not_fit"
            notes.append(
                f"Estimated {human_bytes(vram_total)} of VRAM against {human_bytes(gpu_memory)} available on "
                f"{gpus[0]['name']}."
            )
        elif vram_total > gpu_memory * 0.85:
            risk = "tight"
        else:
            risk = "fits"
    elif ram_total:
        if ram_needed > (ram_available or ram_total * 0.6):
            risk = "tight"
            notes.append(
                f"CPU training: about {human_bytes(ram_needed)} of RAM needed, "
                f"{human_bytes(ram_available or 0)} currently free."
            )
        else:
            risk = "fits_cpu"
    if gpu_memory and ram_total and is_full and exact_params * 4 > gpu_memory:
        warnings.append(
            "Full fine-tuning keeps optimizer state in system RAM or on the GPU; on this machine a "
            "LoRA/QLoRA run is the realistic path."
        )
    if sequence_length > int(config.get("max_position_embeddings") or sequence_length):
        warnings.append(
            f"Sequence length {sequence_length} exceeds the model's context window "
            f"({config.get('max_position_embeddings')})."
        )

    return {
        "labelled": "estimated",
        "method": method,
        "precision": precision,
        "quantization": quantization,
        "parameters": {
            "total": exact_params,
            "trainable": int(trainable),
            "frozen": int(max(0, exact_params - trainable)),
            "source": "weight headers (exact)" if weight_bytes else "derived from config.json (estimated)",
        },
        "memory": {
            "weights": int(weights),
            "gradients": int(gradients),
            "optimizer": int(optimizer),
            "activations": int(activations),
            "overhead_factor": overhead,
            "vram_estimate": vram_total,
            "ram_estimate": ram_needed,
            "gpu_memory_available": gpu_memory,
            "ram_available": ram_available,
        },
        "steps": {
            "per_epoch": int(tokens["tokens"] / tokens_per_step) if tokens_per_step else 0,
            "total": steps,
            "effective_batch_size": effective_batch,
            "tokens_per_step": tokens_per_step,
            "optimizer_updates": steps,
        },
        "data": tokens,
        "disk": {
            "checkpoint_size_estimate": checkpoint_bytes,
            "planned_checkpoints": min(checkpoint_limit, max(1, checkpoints_planned)) if steps else 0,
            "workspace_needed": disk_needed,
        },
        "risk": risk,
        "notes": notes,
        "warnings": warnings,
        "formulas": {
            "weights": "parameters × bytes per parameter",
            "full_finetune_memory": "weights + gradients + AdamW states (m, v, master weights) + activations",
            "adapter_memory": "frozen weights at load precision + adapter gradients and states",
            "activations": "batch × sequence × hidden × layers × precision × 1.6",
        },
    }


def to_summary(plan: dict[str, Any]) -> dict[str, str]:
    """Human readable one-liners for the confirmation step."""
    memory = plan["memory"]
    return {
        "vram": human_bytes(memory["vram_estimate"]),
        "ram": human_bytes(memory["ram_estimate"]),
        "disk": human_bytes(plan["disk"]["workspace_needed"]),
        "steps": str(plan["steps"]["total"]),
        "trainable": f"{plan['parameters']['trainable']:,}",
    }


def dump(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2)
