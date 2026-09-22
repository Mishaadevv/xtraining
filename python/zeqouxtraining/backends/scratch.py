"""From-scratch training backend.

Trains a small GPT-style transformer **from random initialisation** on the
user's dataset. There is no base model to download: the architecture comes from
the job config (`scratch_*` keys) and the vocabulary is learned from the data
itself with a character-level tokenizer built on the fly when no real tokenizer
is available.

This answers a fair question the UI could not answer before: "why can't I train
a model from zero?" — now it can. It is honest about the trade-off: a model
trained from scratch needs far more data than a fine-tune to become useful,
which the wizard says out loud when the method is selected.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import effective_batch_size
from .base import RunContext, RunResult, TrainingBackend
from .common import dump_json, jsonable, supported_kwargs

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


#: The special tokens every character vocabulary starts with. The count matters:
#: a vocabulary learned from data begins after these, and `decode` skips exactly
#: the tokens written as `<...>`.
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
#: File the tokenizer is written to — deliberately not `tokenizer.json`, so a
#: scratch model keeps working while `transformers` never mistakes it for one of
#: its own tokenizers.
TOKENIZER_FILE = "zeqou_tokenizer.json"


class CharTokenizer:
    """A minimal character-level tokenizer.

    Used only when the run has no usable tokenizer (which is the normal case
    when training from scratch). It is deterministic and needs no downloads: the
    vocabulary is the most common characters of the dataset itself.

    It stands in for a real tokenizer everywhere the app needs one, including
    inference, so a model trained from scratch can actually be talked to in the
    Playground instead of only being saved.
    """

    def __init__(self, texts: list[str] | None = None, vocab_size: int = 0, itos: list[str] | None = None):
        if itos is not None:
            self._init_vocab([str(token) for token in itos])
            return
        counts: Counter[str] = Counter()
        for text in texts or []:
            counts.update(text)
        # Most common characters first; the rest map to <unk>.
        room = max(1, int(vocab_size) - len(SPECIAL_TOKENS))
        common = [char for char, _ in counts.most_common(room)]
        self._init_vocab([*SPECIAL_TOKENS, *common])

    def _init_vocab(self, itos: list[str]) -> None:
        # A vocabulary missing the specials (an older run) is still usable.
        for token in reversed(SPECIAL_TOKENS):
            if token not in itos:
                itos.insert(0, token)
        self.itos = itos
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.name = "zeqou-char-level"
        self.vocab_size = len(self.itos)
        self.chat_template = None

    # The ids the generation loop reads off the tokenizer.
    @property
    def pad_token_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @eos_token.setter
    def eos_token(self, value: str) -> None:
        # `transformers` assigns this while loading; the vocabulary decides the id
        # and an unknown value (None) is simply ignored.
        if value and value in self.stoi:
            self._eos_token = value
        elif not hasattr(self, "_eos_token"):
            self._eos_token = "<eos>"

    def __call__(
        self,
        texts: str | list[str],
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        padding: bool | str = False,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]
        rows: list[list[int]] = []
        for text in texts:
            row = [self.stoi.get(char, self.unk_token_id) for char in str(text)]
            if max_length and truncation:
                row = row[:max_length]
            rows.append(row)

        if return_tensors is None:
            return {"input_ids": rows}
        return self._tensors(rows)

    def _tensors(self, rows: list[list[int]]) -> dict[str, Any]:
        """Padded `input_ids` plus a matching attention mask."""
        try:
            import torch  # noqa: PLC0415 - only inference needs tensors
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Tensor input needs PyTorch. Install the ML runtime in Settings → Environment."
            ) from exc

        width = max(1, max((len(row) for row in rows), default=1))
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    # ------------------------------------------------------------- persistence
    def save_pretrained(self, directory: str) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_class": "ZeqouCharTokenizer",
            "model_type": "char-level",
            "vocab": self.itos,
        }
        (target / TOKENIZER_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Reference copies: other tools can read the plain character -> id map,
        # and the config says out loud which tokenizer this folder holds.
        (target / "vocab.json").write_text(
            json.dumps(self.stoi, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (target / "tokenizer_config.json").write_text(
            json.dumps({
                "tokenizer_class": "ZeqouCharTokenizer",
                "model_type": "char-level",
                "vocab_file": TOKENIZER_FILE,
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "model_max_length": 1024,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def has_vocab(cls, directory: str | Path) -> bool:
        """True when a folder holds a character vocabulary written by this class."""
        return (Path(directory) / TOKENIZER_FILE).is_file()

    @classmethod
    def load_from_dir(cls, directory: str | Path) -> "CharTokenizer":
        """Rebuild the tokenizer saved next to a from-scratch run."""
        source = Path(directory) / TOKENIZER_FILE
        if not source.is_file():
            raise FileNotFoundError(f"No character tokenizer found at {source}.")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"'{source.name}' could not be read: {exc}") from exc
        itos = [str(token) for token in (payload.get("vocab") or [])]
        if not itos:
            raise ValueError(f"'{source.name}' contains no vocabulary.")
        return cls(itos=itos)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        for index in ids:
            index = int(index)
            if index < 0 or index >= len(self.itos):
                continue
            token = self.itos[index]
            if skip_special_tokens and token.startswith("<") and token.endswith(">"):
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
            GPT2Config,
            GPT2LMHeadModel,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        config = dict(ctx.config)
        run_dir = Path(ctx.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        # Same device contract as the HF backend: cuda/cpu/both/auto, and a
        # loud failure when CUDA was requested but does not exist.
        requested = str(config.get("device") or "auto").lower()
        cuda_available = torch.cuda.is_available()
        if requested == "cuda" and not cuda_available:
            raise RuntimeError(
                "CUDA was selected but torch sees no GPU. Check nvidia-smi, install "
                "the CUDA torch build, or switch the device to CPU / Auto."
            )
        if requested in ("cuda", "both") and cuda_available:
            device = "cuda"
        elif requested == "cpu":
            device = "cpu"
        else:
            device = "cuda" if cuda_available else "cpu"
        device_note = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
        arch = resolve_architecture(config)
        ctx.emit("training-status", {
            "message": f"Training device: {device.upper()}{device_note}",
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
            tokenizer = CharTokenizer(texts, int(arch["vocab"]))
            ctx.emit("training-status", {
                "message": f"No tokenizer available — learned a character-level one from the dataset ({tokenizer.vocab_size:,} symbols).",
                "phase": "tokenizer",
            })
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "<pad>"
        tokenizer.padding_side = "right"

        # ------------------------------------------------------------- model
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
            encoded = tokenizer(batch["text"], truncation=True, max_length=context_length)
            # Close every sample with <eos> so generation has somewhere to stop.
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos_id, int):
                encoded["input_ids"] = [[*row, eos_id] for row in encoded["input_ids"]]
            return encoded

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
        training_args = TrainingArguments(**supported_kwargs(TrainingArguments.__init__, argument_kwargs))
        total_steps = int(getattr(training_args, "max_steps", 0) or planned_steps)

        # ---------------------------------------------------------- callbacks
        history, state, stop_state = make_history()
        callback = make_progress_callback(ctx, history, state, stop_state, started, total_steps, config)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=[callback],
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

        status = "paused" if stop_state["pause"] else ("stopped" if stop_state["stop"] else "completed")

        metrics = {}
        if getattr(train_result, "metrics", None):
            metrics = {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))}

        final_loss = history["loss"][-1] if history["loss"] else metrics.get("train_loss")

        metadata = {
            "name": config.get("output_name") or run_dir.name,
            "job_id": ctx.job_id,
            "train_mode": "scratch",
            "method": "scratch",
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
        dump_json(run_dir / "training_history.json", history)
        dump_json(run_dir / "training_config.json", config)
        (run_dir / "metadata.json").write_text(
            json.dumps(jsonable(metadata), ensure_ascii=True, indent=2), encoding="utf-8"
        )

        return RunResult(
            status=status,
            final_loss=final_loss,
            total_steps=real_steps,
            history=history,
            output_dir=str(run_dir),
            last_checkpoint=state["last_checkpoint"],
            metrics=metrics,
        )


def make_history() -> tuple[dict[str, list[Any]], dict[str, Any], dict[str, bool]]:
    """Fresh history/state containers shared by both built-in backends."""
    history: dict[str, list[Any]] = {
        "loss": [], "eval_loss": [], "lr": [], "grad_norm": [], "epoch": [], "step": [],
    }
    state: dict[str, Any] = {"steps": 0, "last_checkpoint": None}
    stop_state: dict[str, bool] = {"stop": False, "pause": False}
    return history, state, stop_state


def make_progress_callback(
    ctx: RunContext,
    history: dict[str, list[Any]],
    state: dict[str, Any],
    stop_state: dict[str, bool],
    started: float,
    total_steps: int,
    config: dict[str, Any],
):
    """Build the Trainer callback that streams metrics over the protocol."""
    from transformers import TrainerCallback  # noqa: PLC0415

    outer = ctx

    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, trainer_state, control, **kwargs):
            if outer.requested_stop() or outer.requested_pause():
                stop_state["stop"] = outer.requested_stop()
                stop_state["pause"] = outer.requested_pause()
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
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    gpu_allocated = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
                    gpu_reserved = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
            except Exception:
                pass

            progress = int(min(99.0, (step / max(1, total_steps)) * 100)) if total_steps else 0
            outer.emit("training-progress", {
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
            outer.emit("training-checkpoint", {
                "step": int(trainer_state.global_step),
                "path": str(checkpoint_dir),
                "size_bytes": size,
                "loss": history["loss"][-1] if history["loss"] else None,
                "epoch": round(float(trainer_state.epoch or 0.0), 4),
                "message": f"Checkpoint saved at step {int(trainer_state.global_step)}",
            })
            return control

    return ProgressCallback()
