"""Training configuration: defaults, validation and automatic tuning.

The whole point of ZeqouXTraining is that the user should not have to hand-write
a YAML file. ``auto_configure`` therefore derives a safe starting point from the
detected hardware, the model and the dataset — and returns a *reason* for every
choice so the UI can explain itself instead of guessing silently.
"""

from __future__ import annotations

from typing import Any

METHODS = ("lora", "qlora", "sft", "full", "scratch")
QUANTIZATIONS = ("none", "4bit", "8bit")
PRECISIONS = ("auto", "bf16", "fp16", "fp32")
SCHEDULERS = ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup")

METHOD_INFO: dict[str, dict[str, Any]] = {
    "lora": {
        "label": "LoRA",
        "summary": "Train small adapter matrices, keep the base model frozen.",
        "detail": "Fastest and cheapest option. Produces a small adapter that is "
                  "loaded on top of the base model. Best default for most tasks.",
        "needs_base_model": True,
        "quantized": False,
    },
    "qlora": {
        "label": "QLoRA",
        "summary": "LoRA on top of a 4-bit quantized base model.",
        "detail": "Fits much larger models into limited VRAM. Requires bitsandbytes "
                  "and an NVIDIA GPU. Slightly slower per step than LoRA.",
        "needs_base_model": True,
        "quantized": True,
    },
    "sft": {
        "label": "SFT (supervised fine-tuning)",
        "summary": "Instruction tuning with LoRA adapters over chat-formatted data.",
        "detail": "Same engine as LoRA but applies the model's chat template to "
                  "message-style datasets. Use it for assistant-style data.",
        "needs_base_model": True,
        "quantized": False,
    },
    "full": {
        "label": "Full fine-tuning",
        "summary": "Update every weight of the base model.",
        "detail": "Maximum quality, needs a lot of VRAM and produces a full-size "
                  "model. Optimizer states alone cost roughly two extra copies of "
                  "the model in fp32.",
        "needs_base_model": True,
        "quantized": False,
    },
    "scratch": {
        "label": "From scratch",
        "summary": "Train a fresh model from random weights on your dataset.",
        "detail": "No base model, no downloads: a small transformer is built from "
                  "the architecture settings and trained on the dataset. Useful for "
                  "tiny domain models and experiments; it needs far more data than "
                  "fine-tuning to become good.",
        "needs_base_model": False,
        "quantized": False,
    },
}

DEFAULTS: dict[str, Any] = {
    # --- from-scratch architecture (used only when method == "scratch") -----
    "scratch_size": "tiny",
    "scratch_layers": 4,
    "scratch_hidden": 256,
    "scratch_heads": 4,
    "scratch_vocab": 16000,
    "scratch_ffn": 1024,
    "method": "lora",
    "base_model": "",
    "output_name": "",
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "context_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "auto",
    "quantization": "none",
    "precision": "auto",
    "gradient_checkpointing": True,
    "optimizer": "auto",
    "save_steps": 0,
    "save_total_limit": 3,
    "eval_split": 0.1,
    "logging_steps": 1,
    "max_samples": 0,
    "seed": 42,
    "device": "auto",
    "resume_from_checkpoint": None,
    "dataset": {
        "path": "",
        "format": "auto",
        "mapping": None,
        "hf_id": None,
        "split": "train",
    },
}

# (min_vram_gb, batch_size, grad_accum, context_length, quantization,
#  gradient_checkpointing, lora_r)
_VRAM_TIERS: list[tuple[float, int, int, int, str, bool, int]] = [
    (8.0,    1, 16, 512,  "4bit", True,  8),
    (12.0,   2, 8,  512,  "4bit", True,  16),
    (16.0,   4, 4,  1024, "none", True,  16),
    (24.0,   4, 4,  1024, "none", False, 16),
    (40.0,   8, 2,  2048, "none", False, 32),
    (1000.0, 16, 1, 4096, "none", False, 64),
]


def normalize(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Fill in defaults, coerce types and keep the shape stable."""
    config = dict(DEFAULTS)
    config["dataset"] = dict(DEFAULTS["dataset"])
    raw = raw or {}

    for key, value in raw.items():
        if key == "dataset" and isinstance(value, dict):
            config["dataset"].update(value)
        elif key in DEFAULTS or key in ("job_id", "run_dir", "total_steps"):
            config[key] = value

    for key in ("epochs", "batch_size", "gradient_accumulation", "context_length",
                "lora_r", "lora_alpha", "save_steps", "save_total_limit",
                "logging_steps", "max_samples", "seed"):
        try:
            config[key] = int(config[key])
        except (TypeError, ValueError):
            config[key] = DEFAULTS[key]

    for key in ("learning_rate", "warmup_ratio", "weight_decay", "lora_dropout",
                "eval_split", "max_grad_norm"):
        try:
            config[key] = float(config[key])
        except (TypeError, ValueError):
            config[key] = DEFAULTS[key]

    config["gradient_checkpointing"] = bool(config["gradient_checkpointing"])

    method = str(config.get("method") or "lora").lower()
    config["method"] = method if method in METHODS else "lora"

    quant = str(config.get("quantization") or "none").lower()
    config["quantization"] = quant if quant in QUANTIZATIONS else "none"
    if config["method"] == "qlora":
        config["quantization"] = "4bit"

    precision = str(config.get("precision") or "auto").lower()
    config["precision"] = precision if precision in PRECISIONS else "auto"

    scheduler = str(config.get("lr_scheduler") or "cosine").lower()
    config["lr_scheduler"] = scheduler if scheduler in SCHEDULERS else "cosine"

    config["batch_size"] = max(1, config["batch_size"])
    config["gradient_accumulation"] = max(1, config["gradient_accumulation"])
    config["epochs"] = max(1, config["epochs"])
    config["context_length"] = max(64, config["context_length"])
    config["lora_r"] = max(1, config["lora_r"])
    config["lora_alpha"] = max(1, config["lora_alpha"])
    config["seed"] = max(0, config["seed"])

    return config


def effective_batch_size(config: dict[str, Any]) -> int:
    return max(1, int(config["batch_size"])) * max(1, int(config["gradient_accumulation"]))


def validate(config: dict[str, Any], hardware: dict[str, Any] | None = None,
             backend_support: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    """Report problems before a run starts, with actionable hints."""
    issues: list[dict[str, Any]] = []

    def issue(severity: str, code: str, message: str, hint: str = "", field: str = "") -> None:
        issues.append({"severity": severity, "code": code, "message": message,
                       "hint": hint, "field": field})

    # Training from scratch is the one method that needs no base model.
    if not config.get("base_model") and config.get("method") != "scratch":
        issue("error", "no_base_model",
              "No base model selected.",
              "Pick a Hugging Face model, point at a local model folder, or switch the method to From scratch.", "base_model")

    dataset = config.get("dataset") or {}
    if not dataset.get("path") and not dataset.get("hf_id"):
        issue("error", "no_dataset",
              "No dataset selected.",
              "Import a JSON/JSONL/CSV/TXT file or point at a dataset folder.", "dataset")

    hardware = hardware or {}
    cuda_ready = bool(hardware.get("cuda_ready"))

    if config["method"] == "qlora" and not cuda_ready:
        issue("error", "qlora_requires_cuda",
              "QLoRA needs an NVIDIA GPU with CUDA enabled.",
              "Switch the method to LoRA, or enable CUDA in Settings → Environment.",
              "method")

    if config["quantization"] != "none" and not cuda_ready:
        issue("warning", "quantization_without_cuda",
              f"{config['quantization']} quantization will be ignored without a CUDA device.",
              "Quantization only applies on CUDA (bitsandbytes). Training will use full precision.",
              "quantization")

    if config["method"] == "full" and not cuda_ready:
        issue("warning", "full_ft_on_cpu",
              "Full fine-tuning on CPU will be extremely slow.",
              "Use LoRA/QLoRA on a CUDA device, or reduce the model size.", "method")

    if config["method"] == "scratch":
        size = str(config.get("scratch_size") or "tiny").lower()
        if size not in ("micro", "tiny", "small"):
            issue("warning", "scratch_size_unknown",
                  f"Unknown scratch size '{size}'; the default 'tiny' will be used.",
                  "Sizes: micro, tiny, small.", "scratch_size")
        if not cuda_ready:
            issue("warning", "scratch_on_cpu",
                  "Training from scratch on CPU is slow even for a tiny model.",
                  "Keep the dataset small and the epochs modest on this machine.", "method")
        vocab = int(config.get("scratch_vocab") or 0)
        if vocab and vocab < 1000:
            issue("warning", "scratch_vocab_small",
                  f"A vocabulary of {vocab:,} is very small.",
                  "Vocabularies below 1 000 tokens can only express tiny domains.", "scratch_vocab")

    if config["precision"] in ("fp16", "bf16") and not cuda_ready:
        issue("warning", "precision_ignored",
              f"{config['precision']} requires CUDA and will fall back to fp32 on CPU.",
              "This is handled automatically; no action needed.", "precision")

    if config["precision"] == "bf16" and cuda_ready and not hardware.get("cuda", {}).get("bf16_supported"):
        issue("warning", "bf16_unsupported",
              "This GPU does not support bfloat16.",
              "Switch precision to fp16, or leave it on auto.", "precision")

    if config["batch_size"] > 32:
        issue("warning", "large_batch",
              f"Batch size {config['batch_size']} is unusually large.",
              "Large batches usually need a higher learning rate and more VRAM.", "batch_size")

    if config["learning_rate"] > 1e-2:
        issue("warning", "high_lr",
              f"Learning rate {config['learning_rate']:g} is very high for fine-tuning.",
              "LoRA typically uses 1e-4 – 5e-4, full fine-tuning 1e-5 – 5e-5.",
              "learning_rate")

    if config["learning_rate"] < 1e-7:
        issue("warning", "low_lr",
              "The learning rate is extremely low; training may appear frozen.",
              "LoRA typically uses 1e-4 – 5e-4.", "learning_rate")

    if config["epochs"] > 20:
        issue("warning", "many_epochs",
              f"{config['epochs']} epochs will take a long time and can overfit.",
              "Small datasets rarely need more than 3–5 epochs.", "epochs")

    if config["context_length"] > 8192:
        issue("warning", "long_context",
              f"Context length {config['context_length']} increases memory sharply.",
              "Attention cost grows quadratically with sequence length.", "context_length")

    if config["method"] in ("lora", "qlora", "sft"):
        if config["lora_r"] > 256:
            issue("warning", "high_lora_r",
                  f"LoRA rank {config['lora_r']} is very high and behaves like full fine-tuning.",
                  "Ranks of 8–64 cover most use cases.", "lora_r")
        if config["lora_alpha"] < config["lora_r"] / 4:
            issue("warning", "low_alpha",
                  "LoRA alpha is low relative to rank, which weakens the adapter.",
                  "A common convention is alpha = 2 × rank.", "lora_alpha")

    support = backend_support or {}
    for name, ready in support.items():
        if not ready:
            issue("error", "backend_unavailable",
                  f"The training backend for '{name}' is not available on this machine.",
                  "Install the ML runtime in Settings → Environment.", "method")
            break

    return issues


#: Values called out when automatic configuration is switched off, so the user
#: knows which defaults actually matter for a first run.
_BASELINE_FIELDS = (
    "method", "batch_size", "gradient_accumulation", "learning_rate",
    "context_length", "lora_r", "lora_alpha", "precision", "quantization",
)


def baseline() -> dict[str, Any]:
    """The documented defaults, with no hardware tuning applied.

    The wizard still needs a valid configuration to render, so when the user
    turns automatic configuration off we hand back `DEFAULTS` — and say so,
    rather than passing these off as choices made for this machine.
    """
    config = normalize({})
    reasons = [
        {
            "field": field,
            "value": config[field],
            "reason": "Documented default — automatic configuration is off, "
                      "so this was not tuned to your hardware. Review it before starting.",
        }
        for field in _BASELINE_FIELDS
        if field in config
    ]
    return {"config": config, "reasons": reasons}


def auto_configure(
    hardware: dict[str, Any] | None,
    model_info: dict[str, Any] | None = None,
    dataset_report: dict[str, Any] | None = None,
    *,
    prefer_quality: bool = False,
) -> dict[str, Any]:
    """Derive a safe configuration plus the reasoning behind every value."""
    hardware = hardware or {}
    model_info = model_info or {}
    reasons: list[dict[str, Any]] = []
    patch: dict[str, Any] = {}

    cuda_ready = bool(hardware.get("cuda_ready"))
    devices = hardware.get("cuda", {}).get("devices") or []
    vram_gb = float(devices[0].get("total_memory_mb") or 0) / 1024 if devices else 0.0
    if not devices:
        smi_gpus = hardware.get("gpu", {}).get("gpus") or []
        vram_gb = float(smi_gpus[0].get("memory_total_mb") or 0) / 1024 if smi_gpus else 0.0

    if not cuda_ready:
        patch.update({
            "method": "lora",
            "quantization": "none",
            "precision": "fp32",
            "batch_size": 1,
            "gradient_accumulation": 8,
            "context_length": 256,
            "gradient_checkpointing": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "device": "cpu",
        })
        reasons.append({
            "field": "device",
            "value": "cpu",
            "reason": "No usable CUDA device was detected, so training falls back to CPU. "
                      "Expect it to be very slow — the Hardware screen shows the exact reason.",
        })
        reasons.append({
            "field": "precision",
            "value": "fp32",
            "reason": "CPU training does not use fp16/bf16 in this pipeline.",
        })
        reasons.append({
            "field": "batch_size",
            "value": 1,
            "reason": "A batch of 1 with gradient accumulation keeps memory use predictable on CPU.",
        })
    else:
        tier = next(t for t in _VRAM_TIERS if vram_gb < t[0])
        _, batch, accum, context, quant, checkpointing, rank = tier
        if prefer_quality and vram_gb >= 16:
            quant = "none"

        method = "lora"
        if quant == "4bit":
            method = "lora"  # QLoRA stays an explicit choice; LoRA is the safer default.

        patch.update({
            "method": method,
            "batch_size": batch,
            "gradient_accumulation": accum,
            "context_length": context,
            "quantization": "none" if method != "qlora" else "4bit",
            "gradient_checkpointing": checkpointing,
            "lora_r": rank,
            "lora_alpha": rank * 2,
            "device": "cuda",
        })
        reasons.append({
            "field": "batch_size",
            "value": batch,
            "reason": f"Chosen for {vram_gb:.1f} GB of VRAM ({hardware.get('cuda', {}).get('devices', [{}])[0].get('name', 'GPU')}).",
        })
        reasons.append({
            "field": "gradient_accumulation",
            "value": accum,
            "reason": f"Keeps the effective batch size at {batch * accum} while fitting in memory.",
        })
        reasons.append({
            "field": "context_length",
            "value": context,
            "reason": "A balanced starting point for memory versus long samples.",
        })
        if quant == "4bit":
            reasons.append({
                "field": "method",
                "value": "LoRA",
                "reason": "LoRA is the default because it trains reliably on this GPU tier. "
                          "Switch to QLoRA if the model does not fit.",
            })
        else:
            reasons.append({
                "field": "quantization",
                "value": "none",
                "reason": "Enough VRAM is available to train without quantizing the base model.",
            })
        if checkpointing:
            reasons.append({
                "field": "gradient_checkpointing",
                "value": True,
                "reason": "Trades some speed for a large drop in activation memory.",
            })

    # Precision follows the device and the GPU's capabilities.
    if cuda_ready:
        bf16_ok = bool(hardware.get("cuda", {}).get("bf16_supported"))
        precision = "bf16" if bf16_ok else "fp16"
        patch["precision"] = precision
        reasons.append({
            "field": "precision",
            "value": precision,
            "reason": ("This GPU supports bfloat16, which is the most numerically stable "
                       "half-precision format.") if bf16_ok else
                      ("This GPU does not report bfloat16 support, so fp16 is used with "
                       "loss scaling."),
        })

    patch["optimizer"] = "auto"

    # From-scratch architecture values come from DEFAULTS; nothing here is
    # hardware-tuned because there is no base model to size against.

    # Learning rate depends on how much of the model is actually being trained.
    method = patch.get("method", "lora")
    if method == "full":
        patch["learning_rate"] = 2e-5
        reasons.append({
            "field": "learning_rate",
            "value": 2e-5,
            "reason": "Full fine-tuning updates every weight, so it needs a small learning rate.",
        })
    else:
        patch["learning_rate"] = 2e-4
        reasons.append({
            "field": "learning_rate",
            "value": 2e-4,
            "reason": "Standard LoRA learning rate: adapters are small and can move faster.",
        })

    # Dataset-driven adjustments.
    if dataset_report:
        stats = dataset_report.get("stats") or {}
        max_chars = stats.get("max_chars") or 0
        estimated = max(1, round(max_chars / 4))
        if estimated < patch.get("context_length", 512):
            new_context = max(128, min(patch.get("context_length", 512), _round_up_pow2(estimated)))
            if new_context != patch.get("context_length"):
                patch["context_length"] = new_context
                reasons.append({
                    "field": "context_length",
                    "value": new_context,
                    "reason": f"The longest sample is about {estimated:,} tokens, so a shorter "
                              f"context is enough and saves memory.",
                })
        records = (stats.get("usable") or stats.get("records") or 0)
        if records and records < 200:
            patch["epochs"] = 8
            reasons.append({
                "field": "epochs",
                "value": 8,
                "reason": f"Only {records:,} usable samples, so more epochs help the model converge.",
            })
        if records and records >= 5000:
            patch["epochs"] = 2
            reasons.append({
                "field": "epochs",
                "value": 2,
                "reason": f"{records:,} usable samples is a large dataset; two epochs limit overfitting.",
            })

    # Respect the model's own positional limit.
    max_positions = model_info.get("max_position_embeddings")
    if max_positions and patch.get("context_length", 512) > int(max_positions):
        patch["context_length"] = int(max_positions)
        reasons.append({
            "field": "context_length",
            "value": int(max_positions),
            "reason": "Clamped to the model's maximum supported context length.",
        })

    config = normalize({**DEFAULTS, **patch})
    return {"config": config, "reasons": reasons}


def _round_up_pow2(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return max(64, power)
