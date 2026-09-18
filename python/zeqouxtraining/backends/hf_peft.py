"""Hugging Face + PEFT training backend.

This is the real training engine. It runs ``transformers.Trainer`` with either
PEFT adapters (LoRA, QLoRA, SFT) or full fine-tuning, and emits live metrics
through the event protocol.

Design notes that matter in practice:

* Heavy libraries are imported inside ``run`` so the rest of the app works on a
  machine without PyTorch.
* Sequences are tokenised with truncation only; the data collator pads per batch
  and masks padding out of the loss (``label_pad_token_id=-100``). Pre-padding
  to the full context length wastes memory *and* trains on padding.
* ``TrainingArguments`` field names changed across transformers versions, so the
  keyword set is filtered against the installed signature instead of guessing.
* Stop and pause are cooperative: the flag is checked at every step boundary and
  the run checkpoints before returning, so nothing is lost.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from ..config import effective_batch_size
from .base import RunContext, RunResult, TrainingBackend

# Projections that adapters are normally attached to, in preference order.
_TARGET_CANDIDATES: list[list[str]] = [
    ["q_proj", "k_proj", "v_proj", "o_proj"],
    ["q_proj", "v_proj"],
    ["query", "key", "value", "dense"],
    ["query", "value"],
    ["Wqkv", "out_proj"],
    ["c_attn", "c_proj"],
    ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    ["dense"],
]


def _module_suffixes(model) -> list[str]:
    try:
        import torch.nn as nn  # noqa: PLC0415

        return [name.split(".")[-1] for name, module in model.named_modules()
                if isinstance(module, nn.Linear)]
    except Exception:
        return []


def detect_target_modules(model, configured: Any) -> list[str]:
    """Pick adapter target modules from the model's real module names."""
    if isinstance(configured, list) and configured:
        return [str(name) for name in configured]
    if isinstance(configured, str) and configured not in ("", "auto"):
        return [part.strip() for part in configured.split(",") if part.strip()]

    suffixes = set(_module_suffixes(model))
    for group in _TARGET_CANDIDATES:
        present = [name for name in group if name in suffixes]
        if present:
            return present

    unique: list[str] = []
    for name in _module_suffixes(model):
        if name not in unique:
            unique.append(name)
    return unique[:2] or ["q_proj", "v_proj"]


def _supported_kwargs(target, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments the installed library version does not accept."""
    import inspect  # noqa: PLC0415

    try:
        accepted = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def _load_kwargs(loader, base: dict[str, Any]) -> dict[str, Any]:
    """Handle the torch_dtype → dtype rename across transformers versions."""
    import inspect  # noqa: PLC0415

    try:
        accepted = set(inspect.signature(loader).parameters)
    except (TypeError, ValueError):
        return base
    kwargs = dict(base)
    if "dtype" in accepted and "dtype" not in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype", None)
    elif "torch_dtype" not in accepted:
        kwargs.pop("torch_dtype", None)
    return {key: value for key, value in kwargs.items() if key in accepted and value is not None}


class HuggingFacePeftBackend(TrainingBackend):
    name = "hf-peft"
    label = "Hugging Face + PEFT"
    methods = ("lora", "qlora", "sft", "full")
    requires = ("torch", "transformers", "peft", "accelerate")

    # ------------------------------------------------------------------ setup
    def _resolve_dtype(self, torch, config: dict[str, Any]) -> Any:
        precision = str(config.get("precision") or "auto")
        if precision == "fp32":
            return torch.float32
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        # auto
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _build_texts(self, ctx: RunContext, tokenizer) -> list[str]:
        texts: list[str] = []
        chat_template = getattr(tokenizer, "chat_template", None)
        for sample in ctx.samples:
            if "messages" in sample:
                messages = sample["messages"]
                if chat_template:
                    try:
                        texts.append(tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        ))
                        continue
                    except Exception:
                        pass
                texts.append("\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                ))
            elif "text" in sample:
                texts.append(sample["text"])
        return [text for text in texts if text and text.strip()]

    # -------------------------------------------------------------------- run
    def run(self, ctx: RunContext) -> RunResult:
        import torch  # noqa: PLC0415
        from datasets import Dataset  # noqa: PLC0415
        from peft import (  # noqa: PLC0415
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        config = ctx.config
        method = str(config["method"])
        run_dir = Path(ctx.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctx.emit("training-status", {
            "message": f"Training device: {device.upper()}",
            "phase": "device",
            "device": device,
        })

        # ---------------------------------------------------------- tokenizer
        base_model = str(config["base_model"])
        ctx.emit("training-status", {"message": f"Loading tokenizer for {base_model}", "phase": "tokenizer"})
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=bool(config.get("trust_remote_code")),
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ------------------------------------------------------- quantization
        quantization = str(config.get("quantization") or "none")
        want_quant = quantization in ("4bit", "8bit") and device == "cuda"
        if quantization in ("4bit", "8bit") and device != "cuda":
            ctx.emit("training-status", {
                "message": "Quantization was requested but no CUDA device is available; training in full precision.",
                "phase": "warning",
            })

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(config.get("trust_remote_code")),
            "torch_dtype": self._resolve_dtype(torch, config),
        }

        if want_quant:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            compute_dtype = self._resolve_dtype(torch, config)
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
            ctx.emit("training-status", {
                "message": f"Quantization enabled: {quantization} (bitsandbytes)",
                "phase": "quantization",
            })

        # -------------------------------------------------------------- model
        ctx.emit("training-status", {"message": f"Loading model {base_model}", "phase": "model_loading"})
        model = AutoModelForCausalLM.from_pretrained(
            base_model, **_load_kwargs(AutoModelForCausalLM.from_pretrained, load_kwargs)
        )
        if device == "cpu" and not want_quant:
            model = model.to("cpu")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        ctx.emit("model-info", {
            "params": total_params,
            "loaded_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "device": device,
        })

        # ---------------------------------------------------------------- LoRA
        adapter = method in ("lora", "qlora", "sft")
        if want_quant:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing"))
            )

        if adapter:
            target_modules = detect_target_modules(model, config.get("lora_target_modules"))
            ctx.emit("training-status", {
                "message": f"Attaching LoRA adapters to: {', '.join(target_modules)}",
                "phase": "adapter",
                "target_modules": target_modules,
            })
            lora_config = LoraConfig(
                r=int(config["lora_r"]),
                lora_alpha=int(config["lora_alpha"]),
                lora_dropout=float(config["lora_dropout"]),
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)

        if config.get("gradient_checkpointing"):
            try:
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception as exc:
                ctx.emit("training-status", {
                    "message": f"Gradient checkpointing could not be enabled: {exc}",
                    "phase": "warning",
                })

        try:
            model.config.use_cache = False
        except Exception:
            pass

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ctx.emit("training-status", {
            "message": (f"Trainable parameters: {trainable:,} / {total:,} "
                        f"({100 * trainable / max(1, total):.3f}%)"),
            "phase": "model_ready",
            "trainable_params": trainable,
            "total_params": total,
            "frozen_base": bool(adapter),
        })

        # ------------------------------------------------------------ dataset
        ctx.emit("training-status", {"message": "Preparing dataset", "phase": "dataset"})
        texts = self._build_texts(ctx, tokenizer)
        if not texts:
            raise ValueError("The dataset produced no usable text after mapping and templating.")

        max_samples = int(config.get("max_samples") or 0)
        if max_samples and len(texts) > max_samples:
            ctx.emit("training-status", {
                "message": f"Using the first {max_samples:,} of {len(texts):,} samples (limit set in Advanced).",
                "phase": "dataset",
            })
            texts = texts[:max_samples]

        context_length = int(config["context_length"])
        raw = Dataset.from_dict({"text": texts})

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            # Truncate only: the collator pads per batch and masks the padding.
            return tokenizer(batch["text"], truncation=True, max_length=context_length)

        tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
        tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 1)

        if len(tokenized) == 0:
            raise ValueError(
                f"Every sample was shorter than 2 tokens after tokenisation "
                f"(context length {context_length}). Lower the context length or check the dataset mapping."
            )

        eval_ratio = float(config.get("eval_split") or 0.0)
        eval_dataset = None
        train_dataset = tokenized
        if eval_ratio > 0 and len(tokenized) >= 10:
            split = tokenized.train_test_split(test_size=min(eval_ratio, 0.5), seed=int(config["seed"]))
            train_dataset, eval_dataset = split["train"], split["test"]

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        batch_size = int(config["batch_size"])
        accumulation = int(config["gradient_accumulation"])
        epochs = int(config["epochs"])
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / (batch_size * accumulation)))
        planned_steps = steps_per_epoch * epochs

        ctx.emit("training-status", {
            "message": (f"{len(train_dataset):,} training samples · "
                        f"{steps_per_epoch:,} steps/epoch · {planned_steps:,} total steps"
                        + (f" · {len(eval_dataset):,} evaluation samples" if eval_dataset is not None else "")),
            "phase": "plan",
            "total_steps": planned_steps,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "effective_batch_size": effective_batch_size(config),
        })

        # ------------------------------------------------------ TrainingArguments
        precision = str(config.get("precision") or "auto")
        use_bf16 = device == "cuda" and precision in ("bf16", "auto") and torch.cuda.is_bf16_supported()
        use_fp16 = device == "cuda" and not use_bf16 and precision in ("fp16", "auto")
        if device == "cpu":
            use_bf16 = use_fp16 = False

        optimizer = str(config.get("optimizer") or "auto")
        if optimizer == "auto":
            if want_quant:
                optimizer = "paged_adamw_8bit"
            elif device == "cuda":
                optimizer = "adamw_torch_fused"
            else:
                optimizer = "adamw_torch"

        save_steps = int(config.get("save_steps") or 0) or max(5, planned_steps // 5)
        logging_steps = max(1, int(config.get("logging_steps") or 1))

        strategy = "epoch" if eval_dataset is not None else "no"
        argument_kwargs: dict[str, Any] = {
            "output_dir": str(run_dir),
            "overwrite_output_dir": False,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": max(1, batch_size),
            "gradient_accumulation_steps": accumulation,
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "max_grad_norm": float(config["max_grad_norm"]),
            "warmup_ratio": float(config["warmup_ratio"]),
            "lr_scheduler_type": str(config["lr_scheduler"]),
            "optim": optimizer,
            "logging_steps": logging_steps,
            "logging_first_step": True,
            "save_strategy": "steps",
            "save_steps": save_steps,
            "save_total_limit": max(1, int(config["save_total_limit"])),
            "bf16": use_bf16,
            "fp16": use_fp16,
            "seed": int(config["seed"]),
            "data_seed": int(config["seed"]),
            "report_to": [],
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "gradient_checkpointing": bool(config.get("gradient_checkpointing")),
            "load_best_model_at_end": eval_dataset is not None,
            "metric_for_best_model": "loss",
            "greater_is_better": False,
            "disable_tqdm": True,
        }
        # transformers renamed evaluation_strategy -> eval_strategy in 4.46.
        strategy_key = "eval_strategy"
        try:
            import inspect  # noqa: PLC0415

            params = set(inspect.signature(TrainingArguments.__init__).parameters)
            if "eval_strategy" in params:
                strategy_key = "eval_strategy"
            elif "evaluation_strategy" in params:
                strategy_key = "evaluation_strategy"
            else:
                strategy_key = ""
        except Exception:
            strategy_key = "evaluation_strategy"
        if strategy_key:
            argument_kwargs[strategy_key] = strategy

        training_args = TrainingArguments(**_supported_kwargs(TrainingArguments.__init__, argument_kwargs))
        total_steps = int(getattr(training_args, "max_steps", 0) or planned_steps)

        # ----------------------------------------------------------- callbacks
        history: dict[str, list[Any]] = {"loss": [], "eval_loss": [], "lr": [], "grad_norm": [], "epoch": [], "step": []}
        state = {"steps": 0, "last_checkpoint": None, "last_emit": 0.0, "samples": 0}
        stop_file_state = {"stop": False, "pause": False}

        outer = self

        class ProgressCallback(TrainerCallback):
            def on_step_end(self, args, trainer_state, control, **kwargs):
                if ctx.requested_stop() or ctx.requested_pause():
                    stop_file_state["stop"] = ctx.requested_stop()
                    stop_file_state["pause"] = ctx.requested_pause()
                    control.should_training_stop = True
                return control

            def on_log(self, args, trainer_state, control, logs=None, **kwargs):
                logs = logs or {}
                step = int(trainer_state.global_step or 0)
                state["steps"] = step
                loss = logs.get("loss", logs.get("train_loss"))
                eval_loss = logs.get("eval_loss")
                lr = logs.get("learning_rate")
                grad_norm = logs.get("grad_norm")

                if loss is not None:
                    history["loss"].append(round(float(loss), 6))
                if eval_loss is not None:
                    history["eval_loss"].append(round(float(eval_loss), 6))
                if lr is not None:
                    history["lr"].append(float(lr))
                if grad_norm is not None:
                    history["grad_norm"].append(float(grad_norm))
                history["step"].append(step)
                history["epoch"].append(round(float(trainer_state.epoch or 0.0), 4))

                elapsed = max(0.001, time.time() - started)
                rate = step / elapsed if step else 0.0
                remaining = max(0, total_steps - step)
                eta = int(remaining / rate) if rate > 0 else None

                gpu_allocated = None
                gpu_reserved = None
                if device == "cuda":
                    try:
                        gpu_allocated = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
                        gpu_reserved = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
                    except Exception:
                        pass

                progress = int(min(99.0, (step / max(1, total_steps)) * 100)) if total_steps else 0
                ctx.emit("training-progress", {
                    "progress": progress,
                    "step": step,
                    "total_steps": int(total_steps),
                    "loss": float(loss) if loss is not None else None,
                    "eval_loss": float(eval_loss) if eval_loss is not None else None,
                    "learning_rate": float(lr) if lr is not None else None,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "epoch": round(float(trainer_state.epoch or 0.0), 4),
                    "elapsed_seconds": int(elapsed),
                    "seconds_per_step": round(elapsed / step, 4) if step else None,
                    "steps_per_second": round(rate, 4),
                    "samples_per_second": round(rate * effective_batch_size(config), 3),
                    "eta_seconds": eta,
                    "gpu_memory_allocated_mb": gpu_allocated,
                    "gpu_memory_reserved_mb": gpu_reserved,
                    "message": f"Step {step}/{total_steps}",
                })

            def on_save(self, args, trainer_state, control, **kwargs):
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(trainer_state.global_step)}"
                if not checkpoint_dir.is_dir():
                    return control
                state["last_checkpoint"] = str(checkpoint_dir)
                size = sum(f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file())
                ctx.emit("training-checkpoint", {
                    "step": int(trainer_state.global_step),
                    "path": str(checkpoint_dir),
                    "size_bytes": size,
                    "loss": history["loss"][-1] if history["loss"] else None,
                    "epoch": round(float(trainer_state.epoch or 0.0), 4),
                    "message": f"Checkpoint saved at step {int(trainer_state.global_step)}",
                })
                return control

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=[ProgressCallback()],
        )

        if trainer.state.max_steps:
            total_steps = int(trainer.state.max_steps)

        ctx.emit("training-status", {
            "message": f"Starting {method.upper()} training · {total_steps:,} steps",
            "phase": "training_start",
            "total_steps": total_steps,
            "device": device,
            "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
            "optimizer": optimizer,
            "log_dir": str(run_dir),
        })

        # ---------------------------------------------------------------- train
        resume_from = config.get("resume_from_checkpoint") or None
        if resume_from and not Path(str(resume_from)).exists():
            ctx.emit("training-status", {
                "message": f"Resume checkpoint {resume_from} no longer exists; starting a fresh run.",
                "phase": "warning",
            })
            resume_from = None

        train_result = trainer.train(resume_from_checkpoint=resume_from)

        real_steps = int(getattr(trainer.state, "global_step", 0) or state["steps"])
        run_dir_existing = Path(training_args.output_dir)

        # ---------------------------------------------------------------- save
        ctx.emit("training-status", {"message": "Saving model", "phase": "saving"})
        trainer.save_model(str(run_dir_existing))
        tokenizer.save_pretrained(str(run_dir_existing))

        stopping = stop_file_state["stop"] or stop_file_state["pause"]
        status = "paused" if stop_file_state["pause"] else ("stopped" if stopping else "completed")

        metrics = {}
        if getattr(train_result, "metrics", None):
            metrics = {k: float(v) for k, v in train_result.metrics.items()
                       if isinstance(v, (int, float))}

        final_loss = None
        if history["loss"]:
            final_loss = history["loss"][-1]
        elif metrics.get("train_loss") is not None:
            final_loss = metrics["train_loss"]

        metadata = {
            "name": config.get("output_name") or run_dir_existing.name,
            "job_id": ctx.job_id,
            "project_id": ctx.job_id,
            "train_mode": method,
            "method": method,
            "base_model": base_model,
            "adapter": adapter,
            "quantization": quantization if want_quant else "none",
            "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
            "device": device,
            "lora_r": int(config["lora_r"]) if adapter else None,
            "lora_alpha": int(config["lora_alpha"]) if adapter else None,
            "target_modules": target_modules if adapter else None,
            "trainable_params": trainable,
            "total_params": total,
            "epochs_completed": round(float(getattr(trainer.state, "epoch", 0) or 0), 3),
            "total_steps": real_steps,
            "final_loss": final_loss,
            "status": status,
            "created_at": time.time(),
            "duration_seconds": int(time.time() - started),
            "dataset": config.get("dataset"),
            "dataset_report": {
                "records": config.get("dataset_records"),
                "usable": config.get("dataset_usable"),
            },
        }

        trainer.save_state()
        (run_dir_existing / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        (run_dir_existing / "training_config.json").write_text(
            json.dumps(_jsonable(config), ensure_ascii=True, indent=2), encoding="utf-8"
        )
        (run_dir_existing / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        last_checkpoint = state["last_checkpoint"]
        if not last_checkpoint:
            candidates = [p for p in run_dir_existing.iterdir()
                          if p.is_dir() and p.name.startswith("checkpoint-")]
            if candidates:
                candidates.sort(key=lambda p: int(p.name.split("-")[-1]))
                last_checkpoint = str(candidates[-1])

        return RunResult(
            status=status,
            final_loss=final_loss,
            total_steps=real_steps,
            history=history,
            output_dir=str(run_dir_existing),
            last_checkpoint=last_checkpoint,
            metrics=metrics,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
