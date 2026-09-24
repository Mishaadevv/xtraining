"""The tiny backend: a real neural language model trained in pure Python.

No PyTorch, no NumPy — a context-window embedding model (Bengio-style neural
probabilistic language model) with an actual forward pass, actual backpropagation,
AdamW updates, real checkpoints and real generation. It exists so that the whole
pipeline (dataset → tokenizer → training → checkpoint → resume → inference →
evaluation) is genuinely functional on any machine, including ones with no GPU
and no ML runtime installed.

It is honest about being small: it is a small-CPU trainer, not a substitute for
PyTorch when a real GPU-backed run is possible.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

from .. import datasets as ds
from ..bpe import BPETokenizer, train_bpe
from ..errors import ZxError, from_exception
from ..util import ensure_dir, now_iso, read_json, write_json_atomic
from .base import BackendInfo, TrainSpec

METHOD = "scratch"
LR_MIN_FRACTION = 0.1


class TinyModel:
    """Embedding + hidden layer + softmax over a fixed context window."""

    def __init__(self, vocab_size: int, hidden_size: int, context_length: int, seed: int = 42):
        self.vocab_size = max(4, int(vocab_size))
        self.hidden_size = max(4, int(hidden_size))
        self.context_length = max(1, int(context_length))
        self.input_size = self.context_length * self.hidden_size
        rng = random.Random(seed)

        def matrix(rows: int, cols: int, scale: float) -> list[list[float]]:
            return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

        self.embeddings = matrix(self.vocab_size, self.hidden_size, 0.02)
        self.w1 = matrix(self.hidden_size, self.input_size, 1.0 / math.sqrt(self.input_size))
        self.b1 = [0.0] * self.hidden_size
        self.w2 = matrix(self.vocab_size, self.hidden_size, 1.0 / math.sqrt(self.hidden_size))
        self.b2 = [0.0] * self.vocab_size

    # -- serialisation ----------------------------------------------------- #
    def to_payload(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "context_length": self.context_length,
            "embeddings": self.embeddings,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TinyModel":
        model = cls(payload["vocab_size"], payload["hidden_size"], payload["context_length"])
        model.embeddings = payload["embeddings"]
        model.w1 = payload["w1"]
        model.b1 = payload["b1"]
        model.w2 = payload["w2"]
        model.b2 = payload["b2"]
        return model

    def parameter_count(self) -> int:
        total = self.vocab_size * self.hidden_size
        total += self.hidden_size * self.input_size + self.hidden_size
        total += self.vocab_size * self.hidden_size + self.vocab_size
        return total

    # -- forward / backward ------------------------------------------------ #
    def forward(self, context: list[int]) -> tuple[list[float], dict[str, Any]]:
        dim = self.hidden_size
        x: list[float] = []
        for token in context:
            x.extend(self.embeddings[token])
        hidden_pre = [0.0] * dim
        for row in range(dim):
            weights = self.w1[row]
            total = self.b1[row]
            for column in range(len(x)):
                total += weights[column] * x[column]
            hidden_pre[row] = total
        activation = [math.tanh(value) for value in hidden_pre]
        logits = [0.0] * self.vocab_size
        for row in range(self.vocab_size):
            weights = self.w2[row]
            total = self.b2[row]
            for column in range(dim):
                total += weights[column] * activation[column]
            logits[row] = total
        peak = max(logits)
        exponentials = [math.exp(value - peak) for value in logits]
        total = sum(exponentials) or 1.0
        probabilities = [value / total for value in exponentials]
        return probabilities, {"x": x, "hidden_pre": hidden_pre, "activation": activation, "logits": logits}

    def backward(self, cache: dict[str, Any], target: int, grads: dict[str, Any]) -> float:
        probabilities = cache.get("probs")
        if probabilities is None:  # pragma: no cover - defensive
            raise ZxError(code="internal", message="forward pass cache is incomplete")
        probs: list[float] = probabilities
        loss = -math.log(max(probs[target], 1e-12))
        dz = list(probs)
        dz[target] -= 1.0

        activation = cache["activation"]
        x = cache["x"]
        dim = self.hidden_size
        d_w2 = grads["w2"]
        d_b2 = grads["b2"]
        d_a = [0.0] * dim
        w2 = self.w2
        for row in range(self.vocab_size):
            scale = dz[row]
            if scale == 0.0:
                continue
            weights = w2[row]
            target_row = d_w2[row]
            for column in range(dim):
                value = activation[column]
                target_row[column] += scale * value
                d_a[column] += weights[column] * scale
            d_b2[row] += scale

        d_hidden = [d_a[index] * (1.0 - activation[index] ** 2) for index in range(dim)]
        w1 = self.w1
        d_w1 = grads["w1"]
        d_b1 = grads["b1"]
        d_x = [0.0] * len(x)
        for row in range(dim):
            scale = d_hidden[row]
            if scale == 0.0:
                continue
            weights = w1[row]
            target_row = d_w1[row]
            for column in range(len(x)):
                target_row[column] += scale * x[column]
                d_x[column] += weights[column] * scale
            d_b1[row] += scale

        d_embeddings = grads["embeddings"]
        for position, token in enumerate(cache["context"]):
            row = d_embeddings[token]
            base = position * dim
            for column in range(dim):
                row[column] += d_x[base + column]
        return loss


def _zero_grads(model: TinyModel) -> dict[str, Any]:
    return {
        "embeddings": [[0.0] * model.hidden_size for _ in range(model.vocab_size)],
        "w1": [[0.0] * model.input_size for _ in range(model.hidden_size)],
        "b1": [0.0] * model.hidden_size,
        "w2": [[0.0] * model.hidden_size for _ in range(model.vocab_size)],
        "b2": [0.0] * model.vocab_size,
    }


class AdamW:
    """AdamW with decoupled weight decay, per parameter matrix."""

    def __init__(self, model: TinyModel, learning_rate: float, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step = 0
        self.m = _zero_grads(model)
        self.v = _zero_grads(model)

    def _update_vector(self, parameter: list[float], grad: list[float], m: list[float],
                       v: list[float], lr: float, decay: bool) -> None:
        beta1, beta2 = self.beta1, self.beta2
        bias1 = 1 - beta1 ** self.step
        bias2 = 1 - beta2 ** self.step
        for index in range(len(parameter)):
            value = grad[index]
            m[index] = beta1 * m[index] + (1 - beta1) * value
            v[index] = beta2 * v[index] + (1 - beta2) * value * value
            update = (m[index] / bias1) / (math.sqrt(v[index] / bias2) + self.eps)
            if decay:
                parameter[index] -= lr * (update + self.weight_decay * parameter[index])
            else:
                parameter[index] -= lr * update

    def apply(self, model: TinyModel, grads: dict[str, Any], clip: float, lr: float) -> float:
        # Global gradient norm clipping, exactly as reported in the metrics.
        total = 0.0
        for key in ("embeddings", "w1", "w2"):
            for row in grads[key]:
                total += sum(value * value for value in row)
        for key in ("b1", "b2"):
            total += sum(value * value for value in grads[key])
        norm = math.sqrt(total)
        scale = 1.0 if (clip <= 0 or norm <= clip) else clip / (norm + 1e-6)
        self.step += 1
        lr = lr
        for key, decay in (("embeddings", True), ("w1", True), ("w2", True), ("b1", False), ("b2", False)):
            parameters = getattr(model, key)
            gradient = grads[key]
            moments_m = self.m[key]
            moments_v = self.v[key]
            if isinstance(parameters[0], list):
                for row_index, row in enumerate(parameters):
                    scaled = [value * scale for value in gradient[row_index]]
                    self._update_vector(row, scaled, moments_m[row_index], moments_v[row_index], lr, decay)
            else:
                scaled = [value * scale for value in gradient]
                self._update_vector(parameters, scaled, moments_m, moments_v, lr, decay)
        return norm

    def to_payload(self) -> dict[str, Any]:
        return {"step": self.step, "m": self.m, "v": self.v, "lr": self.lr}

    def load_payload(self, payload: dict[str, Any]) -> None:
        self.step = int(payload.get("step", 0))
        if payload.get("m"):
            self.m = payload["m"]
        if payload.get("v"):
            self.v = payload["v"]


def _lr_at(step: int, total_steps: int, base_lr: float, warmup: int, scheduler: str) -> float:
    if warmup and step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if scheduler in ("constant", "none"):
        return base_lr
    if scheduler == "linear":
        progress = step / max(1, total_steps)
        return base_lr * max(LR_MIN_FRACTION, 1 - progress)
    progress = min(1.0, step / max(1, total_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * max(LR_MIN_FRACTION, cosine)


def _build_token_stream(
    spec: TrainSpec,
    progress: Callable[[dict[str, Any]], None],
) -> tuple[BPETokenizer, list[int], dict[str, Any]]:
    if not spec.dataset_paths:
        raise ZxError(
            code="dataset_missing",
            message="No dataset selected for this run.",
            hint="Import a dataset and select it in the training configuration.",
        )
    texts: list[str] = []
    per_dataset: list[dict[str, Any]] = []
    for index, path in enumerate(spec.dataset_paths):
        limit = 20_000
        extracted = ds.extract_texts(Path(path), spec.mapping, limit=limit, template=spec.template)
        weight = spec.dataset_weights[index] if index < len(spec.dataset_weights) else 1.0
        repeats = max(1, int(round(weight * 1))) if len(spec.dataset_paths) > 1 else 1
        per_dataset.append({
            "path": path,
            "texts": len(extracted),
            "weight": weight,
            "repeats": repeats,
            "characters": sum(len(text) for text in extracted),
        })
        for _ in range(repeats):
            texts.extend(extracted)
    if not texts:
        raise ZxError(
            code="dataset_empty",
            message="The selected datasets produced no usable text.",
            hint="Check the field mapping in the dataset page — the engine found no prompt/response/text columns.",
        )

    if spec.tokenizer_path and Path(spec.tokenizer_path).exists():
        tokenizer = BPETokenizer.load(Path(spec.tokenizer_path))
    else:
        tokenizer = BPETokenizer(train_bpe(texts, vocab_size=spec.vocab_size))

    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text))
        stream.append(tokenizer.vocab_size - 1)
    metadata = {
        "texts": len(texts),
        "tokens": len(stream),
        "vocab_size": tokenizer.vocab_size,
        "datasets": per_dataset,
    }
    return tokenizer, stream, metadata


def _batches(stream: list[int], context: int, batch_size: int, accumulation: int, rng: random.Random):
    """Yield (position-list, progress-size) groups covering one epoch."""
    usable = len(stream) - context - 1
    if usable <= 0:
        return
    positions = list(range(usable))
    rng.shuffle(positions)
    group = batch_size * accumulation
    for start in range(0, len(positions), group):
        yield positions[start:start + group]


def train(spec: TrainSpec, progress: Callable[[dict[str, Any]], None], control: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    out_dir = ensure_dir(Path(spec.output_dir))
    emit = lambda event: progress({**event, "phase": "train"})  # noqa: E731

    tokenizer, stream, data_meta = _build_token_stream(spec, progress)
    emit({"type": "log", "level": "info", "message":
          f"Tokenizer ready: {tokenizer.vocab_size} tokens, corpus {len(stream)} tokens "
          f"from {data_meta['texts']} texts."})

    eval_size = max(1, int(len(stream) * max(0.0, min(0.5, spec.eval_ratio)))) if len(stream) > 200 else 0
    if hasattr(spec, "extra") and spec.extra.get("eval_texts_path"):
        eval_stream: list[int] = []
    else:
        eval_stream = stream[-eval_size:] if eval_size else []
        if eval_size:
            stream = stream[:-eval_size]

    parent_payload = None
    parent_model_dir: Path | None = None
    if spec.parent_checkpoint:
        parent_model_dir = Path(spec.parent_checkpoint)
    elif spec.base_model and Path(spec.base_model).exists() and _is_tiny_model(Path(spec.base_model)):
        parent_model_dir = Path(spec.base_model)

    if spec.method == "scratch" and not parent_model_dir:
        model = TinyModel(tokenizer.vocab_size, spec.hidden_size, spec.context_length, seed=spec.seed)
        lineage_note = "Initialised from random weights (training from scratch)."
    else:
        if not parent_model_dir:
            raise ZxError(
                code="continuation_requires_parent",
                message=f"Method '{spec.method}' needs a base model or a parent checkpoint.",
                hint="Pick a model folder produced by this app, or switch the method to 'Train from scratch'.",
            )
        parent_payload = _load_tiny_payload(parent_model_dir)
        model = TinyModel.from_payload(parent_payload["model"])
        if parent_payload["model"]["vocab_size"] != tokenizer.vocab_size:
            raise ZxError(
                code="tokenizer_mismatch",
                message=(
                    f"Tokenizer mismatch: the parent model uses {parent_payload['model']['vocab_size']} "
                    f"tokens, this run would use {tokenizer.vocab_size}."
                ),
                hint="Reuse the parent tokenizer file (the engine does this automatically when the "
                     "tokenizer file is still next to the model), or start a fresh model.",
            )
        lineage_note = f"Continued from {parent_model_dir.name}."
    emit({"type": "log", "level": "info", "message": lineage_note})

    optimizer = AdamW(model, spec.learning_rate, weight_decay=spec.weight_decay)
    rng = random.Random(spec.seed)
    state = {"step": 0, "epoch": 0, "cursor": 0}
    if spec.resume_from:
        resume_dir = Path(spec.resume_from)
        payload = _load_tiny_payload(resume_dir)
        if payload.get("optimizer"):
            optimizer.load_payload(payload["optimizer"])
        state = payload.get("state", state)
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            rng.setstate(_rng_from_json(rng_state))
        emit({
            "type": "log",
            "level": "info",
            "message": f"Resumed from {resume_dir.name}: step {state.get('step')}, epoch {state.get('epoch')}. "
                       f"Optimizer state restored: {'yes' if payload.get('optimizer') else 'no'}.",
        })

    resumed_step = int(state.get("step") or 0)
    estimated_steps_per_epoch = max(1, (len(stream) - spec.context_length - 1) //
                                    max(1, spec.batch_size * spec.gradient_accumulation))
    if spec.max_steps:
        # When resuming, a step budget means "this many further optimizer steps".
        total_steps = resumed_step + int(spec.max_steps)
    else:
        total_steps = max(1, int(estimated_steps_per_epoch * max(0.01, float(spec.epochs))))

    emit({
        "type": "plan",
        "total_steps": total_steps,
        "resumed_step": resumed_step,
        "additional_steps": max(0, total_steps - resumed_step),
        "vocab_size": tokenizer.vocab_size,
        "parameters": model.parameter_count(),
        "trainable_parameters": model.parameter_count(),
        "context_length": model.context_length,
        "hidden_size": model.hidden_size,
        "tokens": len(stream),
        "eval_tokens": len(eval_stream),
        "resumed_from_step": state.get("step", 0),
    })

    if spec.tokenizer_path is None:
        tokenizer.save(out_dir / "tokenizer.json")
    _write_config(model, out_dir, tokenizer.vocab_size, spec)

    def evaluate_loss() -> float | None:
        if not eval_stream:
            return None
        window = model.context_length
        limit = min(len(eval_stream) - window - 1, 200)
        if limit <= 0:
            return None
        total = 0.0
        for index in range(limit):
            context = eval_stream[index:index + window]
            probabilities, _cache = model.forward(context)
            target = eval_stream[index + window]
            total -= math.log(max(probabilities[target], 1e-12))
        return total / limit

    log_every = max(1, int(spec.logging_every or 10))
    save_every = max(0, int(spec.save_every or 0))
    eval_every = max(0, int(spec.eval_every or 0))
    window = model.context_length
    running_loss = 0.0
    micro = 0
    tokens_seen = 0
    checkpoints: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    stop_reason = "completed"
    epoch = int(state.get("epoch", 0))
    step = int(state.get("step", 0))

    while step < total_steps:
        epoch += 1
        epoch_started = time.time()
        for positions in _batches(stream, window, spec.batch_size, spec.gradient_accumulation, rng):
            if step >= total_steps:
                break
            if control.get("stop"):
                stop_reason = "cancelled"
                break
            if control.get("save_now"):
                control["save_now"] = False
                step = _save_checkpoint(model, optimizer, state, rng, out_dir, tokenizer, step, epoch,
                                        history, spec, checkpoints, forced=True, emit=emit)
            while control.get("pause") and not control.get("stop"):
                time.sleep(0.25)
            learning_rate = _lr_at(step, total_steps, spec.learning_rate, spec.warmup_steps, spec.lr_scheduler)
            grads = _zero_grads(model)
            batch_loss = 0.0
            accumulated = 0
            for position in positions:
                context = stream[position:position + window]
                target = stream[position + window]
                probabilities, cache = model.forward(context)
                cache["probs"] = probabilities
                cache["context"] = context
                loss = model.backward(cache, target, grads)
                batch_loss += loss
                accumulated += 1
                tokens_seen += len(context)
            if accumulated == 0:
                continue
            norm = optimizer.apply(model, grads, spec.max_grad_norm, learning_rate)
            micro += 1
            step += 1
            state.update({"step": step, "epoch": epoch})
            average_loss = batch_loss / accumulated
            running_loss += average_loss

            if micro % log_every == 0 or step == total_steps:
                elapsed = time.time() - started
                throughput = tokens_seen / elapsed if elapsed > 0 else 0.0
                steps_done = max(1, step - resumed_step)
                eta = ((total_steps - step) / (steps_done / elapsed)) if elapsed > 0 else None
                event = {
                    "type": "metrics",
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "loss": round(average_loss, 5),
                    "loss_avg": round(running_loss / micro, 5),
                    "learning_rate": round(learning_rate, 8),
                    "grad_norm": round(norm, 4),
                    "tokens_per_second": round(throughput, 1),
                    "samples_per_second": round(tokens_seen / max(1, window) / elapsed, 2) if elapsed > 0 else None,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1) if eta and eta > 0 else None,
                    "tokens_seen": tokens_seen,
                }
                history.append(event)
                emit(event)

            if eval_every and step % eval_every == 0:
                value = evaluate_loss()
                if value is not None:
                    emit({
                        "type": "metrics",
                        "step": step,
                        "eval_loss": round(value, 5),
                        "perplexity": round(math.exp(min(20, value)), 4),
                    })

            if save_every and step % save_every == 0:
                step = _save_checkpoint(model, optimizer, state, rng, out_dir, tokenizer, step, epoch,
                                        history, spec, checkpoints, emit=emit)

            if stop_reason == "cancelled":
                break
        if stop_reason == "cancelled":
            break
        emit({
            "type": "log",
            "level": "info",
            "message": f"Epoch {epoch} finished in {round(time.time() - epoch_started, 1)}s.",
        })
        if float(spec.epochs) <= epoch and not spec.max_steps:
            break

    final_dir = out_dir / "final"
    ensure_dir(final_dir)
    _save_state(final_dir, model, optimizer, state, rng, tokenizer, spec, history, emit)
    checkpoints.append({
        "name": "final",
        "path": str(final_dir),
        "step": step,
        "epoch": epoch,
        "kind": "final",
    })
    emit({"type": "checkpoint", "name": "final", "path": str(final_dir), "step": step, "kind": "final"})

    eval_loss = evaluate_loss()
    result = {
        "status": "completed" if stop_reason == "completed" else "cancelled",
        "backend": "tiny",
        "method": spec.method,
        "steps": step,
        "epochs": epoch,
        "final_loss": history[-1]["loss"] if history else None,
        "final_eval_loss": round(eval_loss, 5) if eval_loss is not None else None,
        "perplexity": round(math.exp(min(20, eval_loss)), 4) if eval_loss is not None else None,
        "tokens_seen": tokens_seen,
        "parameters": model.parameter_count(),
        "output_dir": str(out_dir),
        "model_dir": str(final_dir),
        "checkpoints": checkpoints,
        "history": history,
        "data": data_meta,
        "duration_seconds": round(time.time() - started, 1),
        "tokenizer": str((out_dir / "tokenizer.json")) if (out_dir / "tokenizer.json").exists()
        else (spec.tokenizer_path or ""),
        "finished_at": now_iso(),
        "lineage_note": lineage_note,
    }
    write_json_atomic(out_dir / "result.json", result)
    emit({"type": "done", "result": result})
    return result


def _is_tiny_model(path: Path) -> bool:
    payload = read_json(Path(path) / "config.json", {}) or {}
    return payload.get("model_type") in ("zx-tiny", "zx_tiny")


def _write_config(model: TinyModel, out_dir: Path, vocab_size: int, spec: TrainSpec) -> None:
    write_json_atomic(out_dir / "config.json", {
        "architectures": ["ZxTinyLM"],
        "model_type": "zx-tiny",
        "vocab_size": vocab_size,
        "hidden_size": model.hidden_size,
        "context_length": model.context_length,
        "num_hidden_layers": 1,
        "torch_dtype": "float32",
        "created_by": "ZeqouXTraining tiny backend",
        "method": spec.method,
        "seed": spec.seed,
    })


def _save_state(directory: Path, model: TinyModel, optimizer: AdamW, state: dict[str, Any],
                rng: random.Random, tokenizer: BPETokenizer, spec: TrainSpec,
                history: list[dict[str, Any]], emit: Callable[[dict[str, Any]], None]) -> None:
    ensure_dir(directory)
    write_json_atomic(directory / "model.json", {
        "model": model.to_payload(),
        "optimizer": optimizer.to_payload(),
        "state": state,
        "rng_state": _rng_to_json(rng),
        "spec": spec.to_dict(),
        "saved_at": now_iso(),
        "backend": "tiny",
    })
    tokenizer.save(directory / "tokenizer.json")
    write_json_atomic(directory / "config.json", {
        "architectures": ["ZxTinyLM"],
        "model_type": "zx-tiny",
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": model.hidden_size,
        "context_length": model.context_length,
        "num_hidden_layers": 1,
        "torch_dtype": "float32",
        "method": spec.method,
        "seed": spec.seed,
    })
    write_json_atomic(directory / "metrics.json", history[-500:])


def _save_checkpoint(model: TinyModel, optimizer: AdamW, state: dict[str, Any], rng: random.Random,
                     out_dir: Path, tokenizer: BPETokenizer, step: int, epoch: int,
                     history: list[dict[str, Any]], spec: TrainSpec,
                     checkpoints: list[dict[str, Any]], emit: Callable[[dict[str, Any]], None],
                     forced: bool = False) -> int:
    name = f"checkpoint-{step}" if not forced else f"checkpoint-{step}"
    directory = out_dir / name
    _save_state(directory, model, optimizer, state, rng, tokenizer, spec, history, emit)
    entry = {
        "name": name,
        "path": str(directory),
        "step": step,
        "epoch": epoch,
        "kind": "checkpoint",
        "created_at": now_iso(),
        "metrics": history[-1] if history else None,
    }
    checkpoints.append(entry)
    emit({"type": "checkpoint", **{k: entry[k] for k in ("name", "path", "step", "kind")}})
    if spec.checkpoint_limit and len([c for c in checkpoints if c["kind"] == "checkpoint"]) > spec.checkpoint_limit:
        removable = [c for c in checkpoints if c["kind"] == "checkpoint"][:-spec.checkpoint_limit]
        for candidate in removable:
            if candidate.get("protected"):
                continue
            try:
                import shutil

                shutil.rmtree(candidate["path"], ignore_errors=True)
                checkpoints.remove(candidate)
                emit({"type": "log", "level": "info",
                      "message": f"Retention policy removed old checkpoint {candidate['name']}."})
            except OSError:
                continue
    return step


def _load_tiny_payload(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "model.json"
    if not path.exists():
        raise ZxError(
            code="checkpoint_missing",
            message=f"No tiny-model checkpoint found in {directory}",
            hint="Select a checkpoint saved by the tiny backend, or switch backends for this model.",
        )
    payload = read_json(path, None)
    if not payload or "model" not in payload:
        raise ZxError(
            code="corrupt_checkpoint",
            message=f"Checkpoint {path.name} is unreadable.",
            hint="Pick another checkpoint — this one is missing its weight payload.",
        )
    return payload


def _rng_to_json(rng: random.Random) -> Any:
    state = rng.getstate()
    return [state[0], list(state[1]), state[2]]


def _rng_from_json(payload: Any) -> Any:
    return (payload[0], tuple(payload[1]), payload[2])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def load_model(model_path: str) -> tuple[TinyModel, BPETokenizer]:
    directory = Path(model_path)
    if (directory / "model.json").exists():
        payload = _load_tiny_payload(directory)
    elif (directory / "final" / "model.json").exists():
        payload = _load_tiny_payload(directory / "final")
        directory = directory / "final"
    else:
        raise ZxError(
            code="model_not_tiny",
            message=f"{directory.name} was not produced by the tiny backend.",
            hint="Load it with the Transformers backend instead (Environment page shows which is active).",
        )
    model = TinyModel.from_payload(payload["model"])
    tokenizer_file = directory / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_file) if tokenizer_file.exists() else BPETokenizer({
        "merges": [], "special_tokens": [], "vocab_size": model.vocab_size,
    })
    return model, tokenizer


def generate(model_path: str, request: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    started = time.time()
    model, tokenizer = load_model(model_path)
    messages = request.get("messages") or []
    prompt = request.get("prompt")
    if messages:
        prompt = "".join(
            f"<|{message.get('role', 'user')}|>\n{message.get('content', '')}\n" for message in messages
        ) + "<|assistant|>\n"
    prompt = prompt or ""
    max_tokens = max(1, min(2048, int(request.get("max_tokens", 128))))
    temperature = float(request.get("temperature", 0.8))
    top_k = int(request.get("top_k", 0) or 0)
    top_p = float(request.get("top_p", 0.95) or 0.95)
    repetition_penalty = float(request.get("repetition_penalty", 1.0) or 1.0)
    seed = request.get("seed")
    rng = random.Random(int(seed) if seed not in (None, "") else None)

    tokens = tokenizer.encode(prompt)
    prompt_tokens = len(tokens)
    if not tokens:
        tokens = [tokenizer.vocab_size - 1]
    window = model.context_length
    produced: list[int] = []
    text = ""
    emit({"type": "start", "prompt_tokens": prompt_tokens, "model": Path(model_path).name})

    for index in range(max_tokens):
        context = tokens[-window:]
        context = [0] * (window - len(context)) + context
        probabilities, _cache = model.forward(context)
        if repetition_penalty != 1.0 and produced:
            for token in set(produced[-64:]):
                probabilities[token] /= repetition_penalty
        if temperature <= 0.01:
            choice = max(range(len(probabilities)), key=lambda token: probabilities[token])
        else:
            adjusted = [value ** (1.0 / temperature) for value in probabilities]
            total = sum(adjusted)
            adjusted = [value / total for value in adjusted]
            if top_k > 0:
                threshold = sorted(adjusted, reverse=True)[min(top_k, len(adjusted)) - 1]
                adjusted = [value if value >= threshold else 0.0 for value in adjusted]
            if 0 < top_p < 1.0:
                order = sorted(range(len(adjusted)), key=lambda token: adjusted[token], reverse=True)
                cumulative = 0.0
                keep: set[int] = set()
                for token in order:
                    cumulative += adjusted[token]
                    keep.add(token)
                    if cumulative >= top_p:
                        break
                adjusted = [value if token in keep else 0.0 for token, value in enumerate(adjusted)]
            total = sum(adjusted) or 1.0
            adjusted = [value / total for value in adjusted]
            r = rng.random()
            cumulative = 0.0
            choice = len(adjusted) - 1
            for token, value in enumerate(adjusted):
                cumulative += value
                if r <= cumulative:
                    choice = token
                    break
        tokens.append(choice)
        produced.append(choice)
        if choice >= 256 + len(tokenizer.merges) and tokenizer.decode([choice]) == "<|end|>":
            break
        text += tokenizer.decode([choice])
        if index % 1 == 0:
            emit({"type": "token", "text": text, "tokens": index + 1})

    elapsed = time.time() - started
    result = {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(produced),
        "latency_seconds": round(elapsed, 3),
        "tokens_per_second": round(len(produced) / elapsed, 2) if elapsed > 0 else None,
        "model": str(model_path),
        "backend": "tiny",
        "finish_reason": "length" if len(produced) >= max_tokens else "stop",
    }
    emit({"type": "done", "result": result})
    return result


def evaluate(model_path: str, dataset_paths: list[str], request: dict[str, Any],
             progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    model, tokenizer = load_model(model_path)
    mapping = request.get("mapping") or {}
    limit = int(request.get("limit", 500))
    texts: list[str] = []
    for path in dataset_paths:
        texts.extend(ds.extract_texts(Path(path), mapping, limit=max(1, limit // max(1, len(dataset_paths))),
                                      template=request.get("template", "chatml")))
    if not texts:
        raise ZxError(
            code="dataset_empty",
            message="Evaluation dataset produced no text.",
            hint="Check the dataset field mapping.",
        )
    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text))
        stream.append(tokenizer.vocab_size - 1)
    window = model.context_length
    usable = len(stream) - window - 1
    if usable <= 0:
        raise ZxError(
            code="dataset_too_small",
            message=f"Only {len(stream)} tokens available; at least {window + 2} are needed.",
            hint="Use a larger evaluation dataset or reduce the model context length.",
        )
    total_loss = 0.0
    evaluated = 0
    correct = 0
    for index in range(usable):
        context = stream[index:index + window]
        target = stream[index + window]
        probabilities, _cache = model.forward(context)
        total_loss -= math.log(max(probabilities[target], 1e-12))
        best = max(range(len(probabilities)), key=lambda token: probabilities[token])
        correct += 1 if best == target else 0
        evaluated += 1
        if evaluated % 100 == 0:
            progress({
                "type": "metrics",
                "phase": "evaluate",
                "tokens": evaluated,
                "total": usable,
                "loss": round(total_loss / evaluated, 5),
            })
        if evaluated >= limit * 50:
            break
    average = total_loss / evaluated
    result = {
        "backend": "tiny",
        "model": str(model_path),
        "metrics": {
            "loss": round(average, 5),
            "perplexity": round(math.exp(min(20, average)), 4),
            "top1_accuracy": round(correct / evaluated, 4),
        },
        "evaluated_tokens": evaluated,
        "datasets": dataset_paths,
        "sequence_length": window,
        "finished_at": now_iso(),
    }
    progress({"type": "done", "result": result})
    return result


class TinyBackend:
    train = staticmethod(train)
    generate = staticmethod(generate)
    evaluate = staticmethod(evaluate)
    info = BackendInfo(
        id="tiny",
        name="Tiny CPU backend (pure Python)",
        description=(
            "A context-window neural language model implemented in pure Python: real forward pass, "
            "real backpropagation, AdamW, checkpoints, resume and streaming generation. "
            "No PyTorch required, runs on any machine, sized for small models and short runs."
        ),
        available=True,
        reason="Ships with the app and needs no external packages.",
        requires=[],
        methods=["scratch", "continued_pretraining", "continued_training", "full_finetune", "sft"],
        supports_gpu=False,
        supports_pause=True,
        supports_from_scratch=True,
        supports_adapters=False,
        supports_preference_training=False,
        speed_note="Pure Python math: expect tens of tokens per second, not thousands.",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TinyBackend available={self.info.available}>"
