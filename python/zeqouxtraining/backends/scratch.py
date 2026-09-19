"""From-scratch training backend.

Trains a small GPT-style transformer **from random initialisation** on the
user's dataset. There is no base model to download: the architecture comes from
the job config (`scratch_*` keys) and the vocabulary is learned from the data
itself with a character/word-level tokenizer built on the fly when no real
tokenizer is available.

This answers a fair question the UI could not before: "why can't I train a
model from zero?" — now it can. It is honest about the trade-off: a model
trained from scratch needs far more data than a fine-tune to become useful,
which the wizard says out loud when the method is selected.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from ..config import effective_batch_size
from .base import RunContext, RunResult, TrainingBackend

# Named architecture presets. Numbers, not vibes: each is small enough to train
# on the hardware this app realistically meets, including CPU-only laptops.
SIZES: dict[str, dict[str, int]] = {
    "micro": {"layers": 2, "hidden": 128, "heads": 2, "ffn": 512, "vocab": 8000},
    "tiny": {"layers": 4, "hidden": 256, "heads": 4, "ffn": 1024, "vocab": 16000},
    "small": {"layers": 6, "hidden": 384, "heads": 6, "ffn": 1536, "vocab": 24000},
}


def resolve_architecture(config: dict[str, Any]) -> dict[str, int]:
    """Merge the named preset with any explicit overrides from the config."""
    size = str(config.get("scratch_size") or "tiny").lower()
    preset = dict(SIZES.get(size, SIZES["tiny"]))
    for key in ("layers", "hidden", "heads", "ffn", "vocab"):
        override = config.get(f"scratch_{key}")
        try:
            value = int(override) if override else 0
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            preset[key] = value
    # Head dimension must divide the hidden size or attention silently breaks.
    preset["heads"] = max(1, min(preset["heads"], preset["hidden"]))
    while preset["hidden"] % preset["heads"] != 0 and preset["heads"] > 1:
        preset["heads"] -= 1
    return preset


class _CharTokenizer:
    """A minimal character-level tokenizer.

    Used only when the run has no usable tokenizer (which is the normal case
    when training from scratch). It is deterministic, needs no downloads, and
    pads with id 0 while keeping the loss masked through the collator.
    """

    def __init__(self, texts: list[str], vocab_size: int):
        from collections import Counter

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text)
        # Most common characters first; the rest map to <unk>.
        common = [char for char, _ in counts.most_common(max(1, vocab_size - 3))]
        self.itos = ["<pad>", "<unk>", "<bos>", *common]
        self.stoi = {char: index for index, char in enumerate(self.itos)}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = None
        self.name = "zeqou-char-level"
        self.vocab_size = len(self.itos)
        self.chat_template = None

    def __call__(self, texts: list[str], truncation: bool = False, max_length: int | None = None):
        ids: list[list[int]] = []
        for text in texts:
            row = [self.stoi.get(char, 1) for char in text]
            if max_length and truncation:
                row = row[:max_length]
            ids.append(row)
        return {"input_ids": ids}

    def save_pretrained(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_class": "ZeqouCharTokenizer",
            "model_type": "char-level",
            "vocab": self.itos,
        }
        (Path(directory) / "zeqou_tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = {"<pad>", "<unk>", "<bos>"}
        out: list[str] = []
        for index in ids:
            if index < 0 or index >= len(self.itos):
                continue
            token = self.itos[index]
            if skip_special_tokens and token in specials:
                continue
            out.append(token)
        return "".join(out)


class ScratchBackend(TrainingBackend):
    """Train a fresh model from random weights — the true "from zero" path."""

    name = "scratch"
    label = "From scratch (small transformer)"
    methods = ("scratch",)
    requires = ("torch", "transformers", "accelerate", "datasets")

    def run(self, ctx: RunContext) -> RunResult:
        import torch  # noqa: PLC0415
        from datasets import Dataset  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        config = dict(ctx.config)
        method = "scratch"
        run_dir = Path(ctx.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        arch = resolve_architecture(config)
        ctx.emit("training-status", {
            "message": f"Training device: {device.upper()}",
            "phase": "device",
            "device": device,
        })

        # ------------------------------------------------------------ texts
        texts: list[str] = []
        for sample in ctx.samples:
            if "messages" in sample:
                texts.append("\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in sample["messages"]
                ))
            elif "text" in sample:
                texts.append(sample["text"])
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            raise ValueError("The dataset produced no usable text after mapping and templating.")

        # -------------------------------------------------------- tokenizer
        # A scratch run normally has no tokenizer of its own; learn one from
        # the data instead of failing or downloading something.
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = None
        if config.get("base_model"):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(config["base_model"]),
                    trust_remote_code=bool(config.get("trust_remote_code")),
                    padding_side="right",
                )
            except Exception:
                tokenizer = None
        if tokenizer is None:
            tokenizer = _CharTokenizer(texts, int(arch["vocab"]))
            ctx.emit("training-status", {
                "message": f"No tokenizer available — learned a character-level one from the dataset ({tokenizer.vocab_size:,} symbols).",
                "phase": "tokenizer",
            })
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "<pad>"
        tokenizer.padding_side = "right"

        # ------------------------------------------------------------- model
        from transformers import GPT2Config, GPT2LMHeadModel  # noqa: PLC0415

        vocab_size = max(int(getattr(tokenizer, "vocab_size", 0) or 0), int(arch["vocab"]))
        model_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=int(config.get("context_length") or 512),
            n_embd=int(arch["hidden"]),
            n_layer=int(arch["layers"]),
            n_head=int(arch["heads"]),
            n_inner=int(arch["ffn"]),
            bos_token_id=2 if vocab_size > 2 else 0,
            eos_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
        )
        ctx.emit("training-status", {
            "message": (
                f"Building a fresh {arch['layers']}-layer transformer "
                f"(hidden {arch['hidden']}, {arch['heads']} heads, vocab {vocab_size:,})"
            ),
            "phase": "model_building",
        })
        model = GPT2LMHeadModel(model_config)

        total_params = sum(p.numel() for p in model.parameters())
        ctx.emit("model-info", {
            "params": total_params,
            "loaded_dtype": "float32",
            "device": device,
        })
        ctx.emit("training-status", {
            "message": f"Random initialisation: {total_params:,} parameters, nothing pre-trained.",
            "phase": "model_ready",
            "trainable_params": total_params,
            "total_params": total_params,
            "frozen_base": False,
        })

        # ----------------------------------------------------------- dataset
        ctx.emit("training-status", {"message": "Preparing dataset", "phase": "dataset"})
        max_samples = int(config.get("max_samples") or 0)
        if max_samples and len(texts) > max_samples:
            texts = texts[:max_samples]

        context_length = int(config.get("context_length") or 512)
        raw = Dataset.from_dict({"text": texts})

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(batch["text"], truncation=True, max_length=context_length)

        tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
        tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 1)
        if len(tokenized) == 0:
            raise ValueError(
                "Every sample was shorter than 2 tokens after tokenisation. "
                "Check the dataset mapping or lower the context length."
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
            "message": (
                f"{len(train_dataset):,} training samples · {steps_per_epoch:,} steps/epoch · "
                f"{planned_steps:,} total steps"
                + (f" · {len(eval_dataset):,} evaluation samples" if eval_dataset is not None else "")
            ),
            "phase": "plan",
            "total_steps": planned_steps,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "effective_batch_size": effective_batch_size(config),
        })

        # ------------------------------------------------- TrainingArguments
        save_steps = int(config.get("save_steps") or 0) or max(5, planned_steps // 5)
        logging_steps = max(1, int(config.get("logging_steps") or 1))
        optimizer = str(config.get("optimizer") or "auto")
        if optimizer == "auto":
            optimizer = "adamw_torch_fused" if device == "cuda" else "adamw_torch"

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
        if eval_dataset is not None:
            argument_kwargs["eval_strategy"] = "epoch"
        training_args = TrainingArguments(**_supported_kwargs(TrainingArguments.__init__, argument_kwargs))
        total_steps = int(getattr(training_args, "max_steps", 0) or planned_steps)

        # ---------------------------------------------------------- callbacks
        history: dict[str, list[Any]] = {
            "loss": [], "eval_loss": [], "lr": [], "grad_norm": [], "epoch": [], "step": [],
        }
        state = {"steps": 0, "last_checkpoint": None}
        stop_file_state = {"stop": False, "pause": False}

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
            "message": f"Starting from-scratch training · {total_steps:,} steps",
            "phase": "training_start",
            "total_steps": total_steps,
            "device": device,
            "precision": "fp32",
            "optimizer": optimizer,
        })

        # ------------------------------------------------------------- train
        resume_from = config.get("resume_from_checkpoint") or None
        if resume_from and not Path(str(resume_from)).exists():
            resume_from = None
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        real_steps = int(getattr(trainer.state, "global_step", 0) or state["steps"])

        # -------------------------------------------------------------- save
        ctx.emit("training-status", {"message": "Saving model", "phase": "saving"})
        trainer.save_model(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))

        status = "paused" if stop_file_state["pause"] else ("stopped" if stop_file_state["stop"] else "completed")

        metrics = {}
        if getattr(train_result, "metrics", None):
            metrics = {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))}

        final_loss = None
        if history["loss"]:
            final_loss = history["loss"][-1]
        elif metrics.get("train_loss") is not None:
            final_loss = metrics["train_loss"]

        metadata = {
            "name": config.get("output_name") or run_dir.name,
            "job_id": ctx.job_id,
            "train_mode": method,
            "method": method,
            "base_model": None,
            "adapter": False,
            "scratch": True,
            "architecture": arch,
            "tokenizer": getattr(tokenizer, "name", "unknown"),
            "precision": "fp32",
            "device": device,
            "trainable_params": total_params,
            "total_params": total_params,
            "total_steps": real_steps,
            "final_loss": final_loss,
            "status": status,
            "created_at": time.time(),
            "duration_seconds": int(time.time() - started),
            "dataset": config.get("dataset"),
        }
        (run_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")
        (run_dir / "training_config.json").write_text(json.dumps(_jsonable(config), ensure_ascii=True, indent=2), encoding="utf-8")
        (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

        last_checkpoint = state["last_checkpoint"]

        return RunResult(
            status=status,
            final_loss=final_loss,
            total_steps=real_steps,
            history=history,
            output_dir=str(run_dir),
            last_checkpoint=last_checkpoint,
            metrics=metrics,
        )


def _supported_kwargs(target, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments the installed transformers version does not accept."""
    import inspect  # noqa: PLC0415

    try:
        accepted = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
