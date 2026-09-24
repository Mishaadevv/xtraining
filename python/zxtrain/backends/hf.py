"""The Transformers backend: real PyTorch training through Transformers + PEFT.

This is the backend that trains real Hugging Face models — full fine-tuning,
LoRA, QLoRA (4-bit), SFT over chat data and continued pretraining over raw text.
It is only reported as available when the required runtime is actually installed,
and every dependency is listed by name when it is not.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from .. import datasets as ds
from ..errors import ZxError, from_exception
from ..hardware import module_available, module_version
from ..util import ensure_dir, now_iso, read_json, write_json_atomic
from .base import BackendInfo, TrainSpec

REQUIRED = ["torch", "transformers", "datasets", "peft"]
TARGET_MODULE_PRESETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


def _missing() -> list[str]:
    return [name for name in REQUIRED if not module_available(name)]


def availability() -> tuple[bool, str, list[str]]:
    missing = _missing()
    if missing:
        return (
            False,
            f"Not installed: {', '.join(missing)}. Install the ML runtime on the Environment page.",
            REQUIRED,
        )
    return True, "", REQUIRED


def _require() -> None:
    missing = _missing()
    if missing:
        raise ZxError(
            code="missing_dependency",
            message=f"The Transformers backend needs: {', '.join(missing)}",
            hint="Open Environment and install the ML runtime into an isolated environment, "
                 "then restart the app so it is detected.",
            context={"missing": missing},
        )


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer  # type: ignore

    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    except Exception as exc:
        raise ZxError(
            code="tokenizer_error",
            message=f"Tokenizer could not be loaded from {path}: {exc}",
            hint="Some tokenizers need sentencepiece or tiktoken. Install the missing package from "
                 "the Environment page.",
        ) from exc


def _load_model(path: str, spec: TrainSpec, for_training: bool):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if spec.quantization in ("int4", "int8") and for_training:
        if not module_available("bitsandbytes"):
            raise ZxError(
                code="missing_dependency",
                message=f"Quantised training ({spec.quantization}) needs bitsandbytes.",
                hint="Install bitsandbytes from the Environment page, or switch the method to LoRA/full fine-tune.",
            )
        from transformers import BitsAndBytesConfig  # type: ignore

        if spec.quantization == "int4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if _bf16_ok() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        if spec.precision == "bf16" and _bf16_ok():
            kwargs["torch_dtype"] = torch.bfloat16
        elif spec.precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        if spec.device not in ("auto", "cpu"):
            kwargs["device_map"] = {"": spec.device} if spec.device.startswith("cuda") else "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except Exception as exc:
        code, _needle, advice = _classify(str(exc))
        raise ZxError(
            code=code,
            message=f"Model could not be loaded: {type(exc).__name__}: {exc}",
            hint=advice,
            detail=str(exc)[-2000:],
        ) from exc
    return model


def _bf16_ok() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _classify(message: str) -> tuple[str, str, str]:
    lowered = message.lower()
    if "out of memory" in lowered or "cuda error" in lowered and "memory" in lowered:
        if "cuda" in lowered:
            return (
                "out_of_memory",
                "cuda out of memory",
                "Reduce the per-device batch size, raise gradient accumulation to keep the effective batch "
                "size, enable gradient checkpointing, or load the model in 4-bit.",
            )
        return "out_of_memory", "out of memory",
        "System RAM was exhausted. Use a smaller model or a shorter sequence length."
    if "trust_remote_code" in lowered or "cannot import" in lowered:
        return ("unsupported_architecture", "custom code",
                "This model needs custom code. Enable custom architectures for this model explicitly after "
                "reviewing its source, or pick another model.")
    if "no module named" in lowered:
        return ("missing_dependency", "missing package",
                "Install the missing package from the Environment page.")
    if "sentencepiece" in lowered or "tiktoken" in lowered:
        return ("tokenizer_error", "tokenizer dependency",
                "Install sentencepiece or tiktoken from the Environment page.")
    if "unsupported dtype" in lowered or "dtype" in lowered and "not supported" in lowered:
        return "unsupported_dtype", "dtype", "Switch the precision to fp32 on this hardware."
    if "no space left" in lowered:
        return "disk_full", "disk full", "Free disk space or change the checkpoint directory."
    return "backend_error", "", "Open the job log for the full traceback."


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #

def _build_hf_dataset(spec: TrainSpec, tokenizer, progress: Callable[[dict[str, Any]], None]):
    import datasets as hf_datasets  # type: ignore

    if not spec.dataset_paths:
        raise ZxError(code="dataset_missing", message="No dataset selected for this run.",
                      hint="Import a dataset and select it in the training configuration.")
    texts: list[str] = []
    for index, path in enumerate(spec.dataset_paths):
        weight = spec.dataset_weights[index] if index < len(spec.dataset_weights) else 1.0
        limit = int(spec.extra.get("max_records", 200_000))
        extracted = ds.extract_texts(Path(path), spec.mapping, limit=limit, template=spec.template)
        if not extracted:
            raise ZxError(
                code="dataset_empty",
                message=f"{Path(path).name} produced no text for chat/completion training.",
                hint="Map the dataset fields (prompt/response or messages) in the dataset page.",
            )
        repeats = max(1, int(round(weight))) if len(spec.dataset_paths) > 1 else 1
        for _ in range(repeats):
            texts.extend(extracted)
    progress({"type": "log", "level": "info",
              "message": f"Prepared {len(texts)} training texts from {len(spec.dataset_paths)} dataset(s)."})

    block_size = spec.sequence_length
    hf_dataset = hf_datasets.Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=block_size, padding=False)

    tokenized = hf_dataset.map(tokenize, batched=True, remove_columns=["text"], desc="tokenising")
    packed = tokenized.map(
        lambda batch: _pack(batch, block_size, tokenizer),
        batched=True,
        remove_columns=tokenized.column_names,
        desc="packing",
    )
    split = packed.train_test_split(test_size=min(0.3, max(0.01, spec.eval_ratio)))
    progress({"type": "log", "level": "info",
              "message": f"Dataset ready: {len(split['train'])} training blocks, {len(split['test'])} eval blocks "
                         f"of {block_size} tokens."})
    return split["train"], split["test"], len(texts)


def _pack(batch: dict[str, Any], block_size: int, tokenizer) -> dict[str, Any]:
    """Concatenate the tokenized texts and cut them into fixed-size blocks."""
    concatenated: list[int] = []
    for ids in batch["input_ids"]:
        concatenated.extend(ids)
        concatenated.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    total = (len(concatenated) // block_size) * block_size
    blocks = [concatenated[i:i + block_size] for i in range(0, total, block_size)]
    return {
        "input_ids": blocks,
        "attention_mask": [[1] * len(block) for block in blocks],
        "labels": [list(block) for block in blocks],
    }


def train(spec: TrainSpec, progress: Callable[[dict[str, Any]], None], control: dict[str, Any]) -> dict[str, Any]:
    _require()
    import torch  # type: ignore
    from transformers import (  # type: ignore
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    started = time.time()
    emit = lambda event: progress({**event, "phase": "train"})  # noqa: E731
    out_dir = ensure_dir(Path(spec.output_dir))
    base_path = spec.resume_from or spec.parent_checkpoint or spec.base_model
    if not base_path:
        raise ZxError(
            code="model_missing",
            message="No base model selected.",
            hint="Import a model first, or use the tiny backend to train from scratch in pure Python.",
        )
    base_path = str(Path(base_path))
    if not Path(base_path).exists():
        raise ZxError(
            code="model_missing",
            message=f"Base model path does not exist: {base_path}",
            hint="Re-import the model; the folder may have been moved or deleted.",
        )

    is_adapter_parent = (Path(base_path) / "adapter_config.json").exists()
    if is_adapter_parent:
        adapter_config = read_json(Path(base_path) / "adapter_config.json", {}) or {}
        base_model = adapter_config.get("base_model_name_or_path")
        if base_model and Path(base_model).exists():
            base_path = base_model
            emit({"type": "log", "level": "info",
                  "message": f"Parent is a LoRA adapter; loading its base model {Path(base_model).name}."})

    tokenizer_path = spec.tokenizer_path or base_path
    tokenizer = _load_tokenizer(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(base_path, spec, for_training=True)
    if hasattr(model, "config"):
        model.config.use_cache = False

    trainable = None
    adapter_note = ""
    if spec.method in ("lora", "qlora", "adapter"):
        if not module_available("peft"):
            raise ZxError(code="missing_dependency", message="LoRA training needs the peft package.",
                          hint="Install peft from the Environment page, or switch to full fine-tuning.")
        from peft import (  # type: ignore
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        if spec.quantization in ("int4", "int8"):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=spec.gradient_checkpointing)
        targets = spec.target_modules or _target_modules(model)
        config = LoraConfig(
            r=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, config)
        adapter_note = f"LoRA r={spec.lora_rank}, alpha={spec.lora_alpha} on {', '.join(targets)}."
        emit({"type": "log", "level": "info", "message": adapter_note})
    elif spec.method == "preference":
        raise ZxError(
            code="unsupported_method",
            message="Preference optimisation (DPO/ORPO/GRPO) needs the TRL runtime.",
            hint="Install trl from the Environment page; the engine will then expose the preference methods.",
        )

    if spec.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model_report = _model_report(model)
    train_dataset, eval_dataset, text_count = _build_hf_dataset(spec, tokenizer, progress)

    report_to: list[str] = []
    cuda = torch.cuda.is_available()
    use_bf16 = spec.precision == "bf16" and cuda and _bf16_ok()
    use_fp16 = (spec.precision == "fp16" and cuda) or (use_bf16 is False and cuda and spec.precision == "auto")
    device = spec.device if spec.device not in ("auto", "") else ("cuda:0" if cuda else "cpu")

    steps_per_epoch = max(1, len(train_dataset) //
                          max(1, spec.batch_size * spec.gradient_accumulation))
    total_steps = spec.max_steps or max(1, int(steps_per_epoch * max(0.01, float(spec.epochs))))

    arguments = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        overwrite_output_dir=True,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=max(1, spec.batch_size),
        gradient_accumulation_steps=spec.gradient_accumulation,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        max_grad_norm=spec.max_grad_norm,
        num_train_epochs=float(spec.epochs),
        max_steps=int(spec.max_steps) if spec.max_steps else -1,
        lr_scheduler_type=spec.lr_scheduler if spec.lr_scheduler in
        ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup")
        else "cosine",
        warmup_steps=spec.warmup_steps,
        optim=spec.optimizer if spec.optimizer in ("adamw_torch", "adamw_hf", "sgd", "adafactor",
                                                   "paged_adamw_8bit") else "adamw_torch",
        logging_steps=max(1, spec.logging_every),
        save_steps=max(1, spec.save_every),
        eval_steps=max(1, spec.eval_every) if spec.eval_every else None,
        eval_strategy="steps" if spec.eval_every else "no",
        save_total_limit=max(1, spec.checkpoint_limit),
        save_safetensors=True,
        bf16=use_bf16,
        fp16=bool(use_fp16 and not use_bf16),
        tf32=bool(cuda and spec.extra.get("tf32", False)),
        gradient_checkpointing=spec.gradient_checkpointing,
        dataloader_num_workers=int(spec.extra.get("workers", 0) or 0),
        dataloader_pin_memory=bool(cuda),
        seed=spec.seed,
        data_seed=spec.seed,
        report_to=report_to,
        logging_dir=str(out_dir / "logs"),
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    emit({
        "type": "plan",
        "total_steps": total_steps,
        "parameters": model_report["parameters"],
        "trainable_parameters": model_report["trainable"],
        "frozen_parameters": model_report["parameters"] - model_report["trainable"],
        "device": device,
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "texts": text_count,
        "train_blocks": len(train_dataset),
        "eval_blocks": len(eval_dataset),
        "resumed_from_step": 0,
    })

    class Bridge(TrainerCallback):
        def on_log(self, args, state, control_obj, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            event: dict[str, Any] = {
                "type": "metrics",
                "step": state.global_step,
                "total_steps": int(state.max_steps or total_steps),
                "epoch": round(float(state.epoch or 0), 3),
                "elapsed_seconds": round(time.time() - started, 1),
            }
            for source, target in (("loss", "loss"), ("eval_loss", "eval_loss"),
                                   ("learning_rate", "learning_rate"),
                                   ("grad_norm", "grad_norm"), ("train_samples_per_second", "samples_per_second")):
                if source in logs and logs[source] is not None:
                    event[target] = float(logs[source])
            if event.get("eval_loss"):
                event["perplexity"] = round(math.exp(min(20.0, float(event["eval_loss"]))), 4)
            if "loss" in event and event["step"]:
                event["tokens_seen"] = int(state.global_step * spec.batch_size *
                                          spec.gradient_accumulation * spec.sequence_length)
            emit(event)

        def on_save(self, args, state, control_obj, **kwargs):  # noqa: ANN001
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            emit({"type": "checkpoint", "name": checkpoint.name, "path": str(checkpoint),
                  "step": state.global_step, "kind": "checkpoint"})

        def on_train_end(self, args, state, control_obj, **kwargs):  # noqa: ANN001
            emit({"type": "log", "level": "info", "message": "Trainer finished; exporting the model."})

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        data_collator=collator,
        callbacks=[Bridge()],
    )

    resume_checkpoint = None
    if spec.resume_from:
        candidate = Path(spec.resume_from)
        if (candidate / "trainer_state.json").exists():
            resume_checkpoint = str(candidate)
        else:
            hf_dir = out_dir / "hf"
            if (hf_dir / "trainer_state.json").exists():
                resume_checkpoint = str(hf_dir)
    if resume_checkpoint:
        emit({"type": "log", "level": "info",
              "message": f"Exact resume from {Path(resume_checkpoint).name}: optimizer, scheduler and RNG state "
                         f"are restored by the Trainer."})

    result: dict[str, Any] = {}
    try:
        train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
        result["train_runtime"] = round(float(train_output.metrics.get("train_runtime", 0.0)), 2)
        result["final_loss"] = train_output.metrics.get("train_loss")
    except Exception as exc:
        code, _needle, advice = _classify(str(exc))
        raise ZxError(
            code=code,
            message=f"Training failed: {type(exc).__name__}: {exc}",
            hint=advice,
            detail=str(exc)[-4000:],
            context={"step": getattr(trainer.state, "global_step", None)},
        ) from exc

    final_dir = out_dir / "final"
    ensure_dir(final_dir)
    if spec.method in ("lora", "qlora", "adapter"):
        model.save_pretrained(str(final_dir))
        emit({"type": "log", "level": "info", "message": f"Adapter saved to {final_dir}"})
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    eval_metrics: dict[str, Any] = {}
    try:
        if len(eval_dataset):
            eval_metrics = {key: float(value) for key, value in trainer.evaluate().items()
                            if isinstance(value, (int, float))}
    except Exception as exc:  # pragma: no cover - evaluation is best effort
        emit({"type": "log", "level": "warning", "message": f"Post-training evaluation failed: {exc}"})

    write_json_atomic(final_dir / "zxtrain-run.json", {
        "job_id": spec.job_id,
        "backend": "hf",
        "method": spec.method,
        "base_model": base_path,
        "adapter_note": adapter_note,
        "spec": spec.to_dict(),
        "metrics": eval_metrics,
        "finished_at": now_iso(),
    })

    result.update({
        "status": "completed",
        "backend": "hf",
        "method": spec.method,
        "steps": int(trainer.state.global_step),
        "epochs": round(float(trainer.state.epoch or 0), 3),
        "perplexity": round(math.exp(min(20.0, eval_metrics["eval_loss"])), 4) if eval_metrics.get("eval_loss") else None,
        "eval_loss": eval_metrics.get("eval_loss"),
        "parameters": model_report["parameters"],
        "trainable_parameters": model_report["trainable"],
        "output_dir": str(out_dir),
        "model_dir": str(final_dir),
        "checkpoints": _list_hf_checkpoints(out_dir),
        "duration_seconds": round(time.time() - started, 1),
        "data": {"texts": text_count, "train_blocks": len(train_dataset)},
        "finished_at": now_iso(),
        "lineage_note": f"Trained with the Transformers backend on {Path(base_path).name}.",
    })
    write_json_atomic(out_dir / "result.json", result)
    emit({"type": "done", "result": result})
    return result


def _list_hf_checkpoints(out_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((out_dir / "hf").glob("checkpoint-*")):
        if path.is_dir():
            step = path.name.split("-")[-1]
            entries.append({
                "name": path.name,
                "path": str(path),
                "step": int(step) if step.isdigit() else None,
                "kind": "checkpoint",
                "has_optimizer": (path / "optimizer.pt").exists(),
                "has_scheduler": (path / "scheduler.pt").exists(),
                "has_rng": (path / "rng_state.pth").exists(),
            })
    return entries


def _model_report(model) -> dict[str, Any]:
    try:
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return {"parameters": int(parameters), "trainable": int(trainable)}
    except Exception:  # pragma: no cover - defensive
        return {"parameters": 0, "trainable": 0}


def _target_modules(model) -> list[str]:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type in TARGET_MODULE_PRESETS:
        return TARGET_MODULE_PRESETS[model_type]
    linear_names: set[str] = set()
    try:
        import torch  # type: ignore
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_names.add(name.split(".")[-1])
    except Exception:  # pragma: no cover - defensive
        pass
    preferred = [name for name in linear_names if name in
                 ("q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "query_key_value")]
    return preferred or sorted(linear_names)[:4] or ["q_proj", "v_proj"]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def generate(model_path: str, request: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    _require()
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # type: ignore

    started = time.time()
    path = Path(model_path)
    adapter_config = read_json(path / "adapter_config.json", {}) or {}
    base_for_adapter = adapter_config.get("base_model_name_or_path")
    load_path = model_path
    if adapter_config and base_for_adapter and Path(base_for_adapter).exists():
        load_path = base_for_adapter

    tokenizer = _load_tokenizer(load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if _bf16_ok() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_config:
        if not module_available("peft"):
            raise ZxError(code="missing_dependency", message="Loading a LoRA adapter needs the peft package.",
                          hint="Install peft from the Environment page.")
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    messages = request.get("messages") or []
    prompt = request.get("prompt")
    if messages:
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "".join(f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}\n" for m in messages)
            prompt += "<|assistant|>\n"
    prompt = prompt or ""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation = {
        "max_new_tokens": max(1, min(4096, int(request.get("max_tokens", 256)))),
        "temperature": float(request.get("temperature", 0.8)),
        "top_p": float(request.get("top_p", 0.95) or 0.95),
        "top_k": int(request.get("top_k", 0) or 0),
        "repetition_penalty": float(request.get("repetition_penalty", 1.0) or 1.0),
        "do_sample": float(request.get("temperature", 0.8)) > 0.01,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if request.get("stop"):
        generation["stop_strings"] = request["stop"]
        generation["tokenizer"] = tokenizer
    if request.get("seed") not in (None, ""):
        torch.manual_seed(int(request["seed"]))

    emit({"type": "start", "prompt_tokens": int(inputs["input_ids"].shape[-1]), "model": path.name})

    import threading

    def run():
        with torch.no_grad():
            model.generate(**inputs, **generation)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    text = ""
    produced = 0
    for chunk in streamer:
        text += chunk
        produced += 1
        emit({"type": "token", "text": text, "tokens": produced})
    thread.join(timeout=5)

    elapsed = time.time() - started
    result = {
        "text": text,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "completion_tokens": produced,
        "latency_seconds": round(elapsed, 3),
        "tokens_per_second": round(produced / elapsed, 2) if elapsed > 0 else None,
        "model": str(model_path),
        "backend": "hf",
        "device": str(getattr(model, "device", "unknown")),
        "finish_reason": "stop",
    }
    emit({"type": "done", "result": result})
    return result


def evaluate(model_path: str, dataset_paths: list[str], request: dict[str, Any],
             progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    _require()
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if _bf16_ok() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    texts: list[str] = []
    for path in dataset_paths:
        texts.extend(ds.extract_texts(Path(path), request.get("mapping") or {},
                                     limit=int(request.get("limit", 200)),
                                     template=request.get("template", "chatml")))
    if not texts:
        raise ZxError(code="dataset_empty", message="Evaluation dataset produced no text.",
                      hint="Check the dataset field mapping.")

    block = int(request.get("sequence_length", 512))
    effective = min(block, int(getattr(model.config, "max_position_embeddings", block) or block))
    batches = 0
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=effective)["input_ids"]
        if ids.shape[-1] < 2:
            continue
        ids = ids.to(model.device)
        with torch.no_grad():
            output = model(ids, labels=ids)
        total_loss += float(output.loss) * (ids.shape[-1] - 1)
        total_tokens += int(ids.shape[-1] - 1)
        batches += 1
        if batches % 10 == 0:
            progress({"type": "metrics", "phase": "evaluate", "batches": batches,
                      "loss": round(total_loss / max(1, total_tokens), 5)})
    if not total_tokens:
        raise ZxError(code="dataset_too_small", message="No evaluable sequences were produced.",
                      hint="Use longer texts or a larger evaluation dataset.")
    average = total_loss / total_tokens
    result = {
        "backend": "hf",
        "model": str(model_path),
        "metrics": {
            "loss": round(average, 5),
            "perplexity": round(math.exp(min(20.0, average)), 4),
        },
        "evaluated_tokens": total_tokens,
        "datasets": dataset_paths,
        "finished_at": now_iso(),
    }
    progress({"type": "done", "result": result})
    return result


def merge_adapter(adapter_path: str, base_model: str | None, destination: str,
                  progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    _require()
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    adapter_config = read_json(Path(adapter_path) / "adapter_config.json", {}) or {}
    base = base_model or adapter_config.get("base_model_name_or_path")
    if not base or not Path(base).exists():
        raise ZxError(
            code="model_missing",
            message="The adapter's base model is not available locally.",
            hint="Import the base model the adapter was trained on, then merge again.",
        )
    progress({"type": "log", "level": "info", "message": f"Merging adapter into {Path(base).name}."})
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    target = ensure_dir(Path(destination))
    merged.save_pretrained(str(target), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(str(target))
    result = {"output": str(target), "base_model": base, "adapter": adapter_path, "finished_at": now_iso()}
    progress({"type": "done", "result": result})
    return result


def quantize_bnb(model_path: str, destination: str, bits: int, progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Real 8-bit/4-bit quantisation with bitsandbytes, saved as a loadable model."""
    _require()
    if not module_available("bitsandbytes"):
        raise ZxError(code="missing_dependency", message="Quantisation to int4/int8 needs bitsandbytes.",
                      hint="Install bitsandbytes from the Environment page.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    progress({"type": "log", "level": "info", "message": f"Quantising to int{bits} with bitsandbytes."})
    config = BitsAndBytesConfig(
        load_in_4bit=bits == 4, load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4" if bits == 4 else None,
    ) if bits == 4 else BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=config, device_map="auto")
    target = ensure_dir(Path(destination))
    model.save_pretrained(str(target), safe_serialization=True)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(str(target))
    result = {"output": str(target), "bits": bits, "backend": "bitsandbytes", "finished_at": now_iso()}
    progress({"type": "done", "result": result})
    return result


class HFBackend:
    train = staticmethod(train)
    generate = staticmethod(generate)
    evaluate = staticmethod(evaluate)

    def __init__(self) -> None:
        available, reason, requires = availability()
        self.info = BackendInfo(
            id="hf",
            name="Transformers + PEFT (PyTorch)",
            description=(
                "Full fine-tuning, LoRA, QLoRA, SFT and continued pretraining for real Hugging Face "
                "models, including resume with optimizer/scheduler/RNG state and adapter merging."
            ),
            available=available,
            reason=reason or "PyTorch runtime detected.",
            requires=requires,
            methods=["full_finetune", "lora", "qlora", "sft", "continued_pretraining",
                     "continued_training", "adapter"],
            supports_gpu=True,
            supports_pause=False,
            supports_continue=True,
            supports_from_scratch=False,
            supports_adapters=True,
            supports_preference_training=module_available("trl"),
            speed_note=(
                f"torch {module_version('torch') or 'n/a'}, transformers "
                f"{module_version('transformers') or 'n/a'}"
            ),
        )
