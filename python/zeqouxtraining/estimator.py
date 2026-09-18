"""VRAM requirement estimation.

This is a deliberately transparent heuristic, not a measurement. It exists to
stop obviously-doomed runs before they start, and it reports its own error bars
so the UI can say "estimate" instead of pretending to know.

Model:

    weights        = params × bytes_per_param(precision, quantization)
    gradients      = trainable_params × bytes_per_param(precision)        (fp32 on CPU)
    optimizer      = trainable_params × 8 for AdamW (two fp32 moments)
    activations    ≈ batch × seq × layers × hidden × k                    (residual/CUDA held)

``k`` is the activation factor. It depends on attention implementation,
gradient checkpointing and the model's MLP ratio, so it is tuned to a practical
range rather than derived exactly. Gradient checkpointing cuts it sharply.
"""

from __future__ import annotations

from typing import Any

# Bytes per stored parameter for each weight format.
_WEIGHT_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "8bit": 1.0,
    "4bit": 0.5,
}

# Activation factor (bytes per element of the per-layer activation budget).
# fp16 activations are the norm for training; checkpointing recomputes them.
_ACTIVATION_FACTOR = {
    (False, False): 6.0,   # no checkpointing
    (True, False): 1.6,    # gradient checkpointing on
    (False, True): 2.4,    # flash/sdpa attention on
    (True, True): 1.0,
}

_OVERHEAD_MB = 700.0  # CUDA context, cuDNN/cuBLAS workspaces, fragmentation


def weight_bytes(precision: str, quantization: str) -> float:
    if quantization == "4bit":
        return _WEIGHT_BYTES["4bit"]
    if quantization == "8bit":
        return _WEIGHT_BYTES["8bit"]
    return _WEIGHT_BYTES.get(precision, 2.0)


def estimate(
    config: dict[str, Any],
    model_info: dict[str, Any] | None = None,
    available_vram_mb: float | None = None,
) -> dict[str, Any]:
    model_info = model_info or {}
    fields = model_info.get("fields") or {}
    params = model_info.get("params")

    if not params:
        return {
            "available": False,
            "reason": "The model's parameter count could not be determined, so VRAM cannot be estimated.",
            "estimated_total_mb": None,
            "verdict": "unknown",
        }

    params = int(params)
    method = str(config.get("method") or "lora")
    precision = str(config.get("precision") or "auto")
    if precision == "auto":
        precision = "bf16"
    quantization = str(config.get("quantization") or "none")

    batch = max(1, int(config.get("batch_size") or 1))
    seq = max(64, int(config.get("context_length") or 512))
    layers = int(fields.get("num_hidden_layers") or 0) or 1
    hidden = int(fields.get("hidden_size") or 0) or 1
    accumulation = max(1, int(config.get("gradient_accumulation") or 1))
    checkpointing = bool(config.get("gradient_checkpointing"))

    # --- weights ---------------------------------------------------------- #
    wbytes = weight_bytes(precision, quantization)
    weights_mb = params * wbytes / (1024 ** 2)

    # --- trainable parameters --------------------------------------------- #
    if method in ("lora", "qlora", "sft"):
        rank = max(1, int(config.get("lora_r") or 16))
        # q_proj/v_proj/(k_proj,o_proj,gate,up,down) ≈ 1.6 attention projections
        target_count = 1.6
        trainable = target_count * layers * 2 * hidden * rank
    else:
        trainable = params

    grad_bytes = 2.0 if precision in ("fp16", "bf16") else 4.0
    gradients_mb = trainable * grad_bytes / (1024 ** 2)
    optimizer_mb = trainable * 8.0 / (1024 ** 2)

    # --- activations ------------------------------------------------------ #
    key = (checkpointing, False)
    factor = _ACTIVATION_FACTOR[key]
    activations_bytes = batch * seq * layers * hidden * factor
    # Accumulated micro-batches only keep the graph of the current one.
    activations_mb = activations_bytes / (1024 ** 2)

    total_mb = weights_mb + gradients_mb + optimizer_mb + activations_mb + _OVERHEAD_MB

    # Activation estimates are the least reliable part; report a range.
    low_mb = total_mb - activations_mb * 0.45
    high_mb = total_mb + activations_mb * 0.60

    verdict = "unknown"
    headroom_mb = None
    if available_vram_mb:
        headroom_mb = available_vram_mb - total_mb
        if total_mb <= available_vram_mb * 0.75:
            verdict = "fits"
        elif total_mb <= available_vram_mb * 0.95:
            verdict = "tight"
        else:
            verdict = "exceeds"

    suggestions: list[str] = []
    if verdict in ("tight", "exceeds"):
        if method not in ("qlora",) and quantization == "none":
            suggestions.append("Switch to QLoRA (4-bit) to shrink the base model by roughly 4×.")
        if not checkpointing:
            suggestions.append("Enable gradient checkpointing to cut activation memory.")
        if batch > 1:
            suggestions.append(f"Lower the batch size from {batch} to {max(1, batch // 2)} and "
                               f"raise gradient accumulation from {accumulation} to "
                               f"{accumulation * 2} to keep the effective batch size.")
        if seq > 512:
            suggestions.append(f"Reduce context length from {seq} to 512.")
        if method in ("lora", "qlora", "sft") and int(config.get("lora_r") or 16) > 16:
            suggestions.append("Lower the LoRA rank; it also lowers optimizer state.")

    return {
        "available": True,
        "params": params,
        "params_exact": bool(model_info.get("params_exact")),
        "trainable_params": int(trainable),
        "weights_mb": round(weights_mb, 1),
        "gradients_mb": round(gradients_mb, 1),
        "optimizer_mb": round(optimizer_mb, 1),
        "activations_mb": round(activations_mb, 1),
        "overhead_mb": _OVERHEAD_MB,
        "estimated_total_mb": round(total_mb, 1),
        "range_low_mb": round(max(0.0, low_mb), 1),
        "range_high_mb": round(high_mb, 1),
        "available_vram_mb": available_vram_mb,
        "headroom_mb": round(headroom_mb, 1) if headroom_mb is not None else None,
        "verdict": verdict,
        "suggestions": suggestions,
        "assumptions": {
            "precision": precision,
            "quantization": quantization,
            "batch_size": batch,
            "context_length": seq,
            "gradient_checkpointing": checkpointing,
            "gradient_accumulation": accumulation,
            "note": "Approximate. Attention kernels, the MLP ratio and fragmentation all shift "
                    "the real number by a meaningful margin.",
        },
    }
