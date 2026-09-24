"""Job execution: pre-flight validation, the run loop, and recovery.

The engine runs every long operation in its own process. The Node side starts
`python -m zxtrain.cli run <spec.json>`; this module owns the job directory
contract:

    <workspace>/jobs/<job_id>/
        spec.json      input written by the UI (never modified here)
        status.json    live state, updated on every meaningful event
        events.jsonl   complete event stream (logs, metrics, checkpoints)
        metrics.jsonl  metric events only, for charts
        control.json   written by the UI: pause / stop / save_now
        lineage.json   parent → child relationships for this run
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import estimator
from .backends import registry
from .backends.base import TrainSpec
from .datasets import inspect_dataset
from .errors import ZxError, from_exception
from .hardware import detect as detect_hardware
from .models import detect_format, inspect as inspect_model
from .util import (
    append_jsonl,
    ensure_dir,
    human_bytes,
    now_iso,
    read_json,
    write_json_atomic,
)

RUNNABLE_KINDS = ("train", "evaluate", "generate", "merge", "quantize", "tokenize", "dataset_clean")


# --------------------------------------------------------------------------- #
# Spec handling
# --------------------------------------------------------------------------- #

def build_spec(request: dict[str, Any], job_dir: Path) -> TrainSpec:
    spec = TrainSpec(
        job_id=str(request.get("job_id") or job_dir.name),
        output_dir=Path(request.get("output_dir") or (job_dir / "output")),
        method=str(request.get("method") or "lora"),
        base_model=request.get("base_model"),
        parent_checkpoint=request.get("parent_checkpoint"),
        resume_from=request.get("resume_from"),
        dataset_paths=[str(path) for path in request.get("dataset_paths") or []],
        dataset_weights=[float(weight) for weight in request.get("dataset_weights") or []],
        mapping=request.get("mapping") or {},
        template=str(request.get("template") or "chatml"),
        tokenizer_path=request.get("tokenizer_path"),
        sequence_length=int(request.get("sequence_length") or 256),
        batch_size=int(request.get("batch_size") or 1),
        gradient_accumulation=int(request.get("gradient_accumulation") or 1),
        epochs=float(request.get("epochs") or 1.0),
        max_steps=int(request.get("max_steps") or 0),
        learning_rate=float(request.get("learning_rate") or 3e-4),
        lr_scheduler=str(request.get("lr_scheduler") or "cosine"),
        warmup_steps=int(request.get("warmup_steps") or 0),
        weight_decay=float(request.get("weight_decay") or 0.0),
        optimizer=str(request.get("optimizer") or "adamw"),
        max_grad_norm=float(request.get("max_grad_norm") or 1.0),
        precision=str(request.get("precision") or "fp32"),
        quantization=str(request.get("quantization") or "none"),
        lora_rank=int(request.get("lora_rank") or 8),
        lora_alpha=int(request.get("lora_alpha") or 16),
        lora_dropout=float(request.get("lora_dropout") or 0.05),
        target_modules=list(request.get("target_modules") or []),
        gradient_checkpointing=bool(request.get("gradient_checkpointing")),
        seed=int(request.get("seed") or 42),
        eval_every=int(request.get("eval_every") or 50),
        save_every=int(request.get("save_every") or 100),
        logging_every=int(request.get("logging_every") or 10),
        checkpoint_limit=int(request.get("checkpoint_limit") or 3),
        device=str(request.get("device") or "auto"),
        eval_ratio=float(request.get("eval_ratio") or 0.05),
        hidden_size=int(request.get("hidden_size") or 32),
        context_length=int(request.get("context_length") or 8),
        vocab_size=int(request.get("vocab_size") or 512),
        extra=request.get("extra") or {},
    )
    spec.extra.setdefault("resumed_step", 0)
    return spec


# --------------------------------------------------------------------------- #
# Pre-flight validation
# --------------------------------------------------------------------------- #

def prepare(request: dict[str, Any]) -> dict[str, Any]:
    """Validate a proposed run and return the plan plus everything the UI must show."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    hardware = detect_hardware([request.get("workspace") or str(Path.home())])

    def check(name: str, ok: bool, message: str, level: str = "error", hint: str = "") -> None:
        checks.append({"name": name, "ok": ok, "message": message, "level": level, "hint": hint})
        if not ok:
            if level == "error":
                errors.append(message)
            else:
                warnings.append(message)

    backend_id = request.get("backend")
    backend = registry.resolve(backend_id)
    method = str(request.get("method") or "lora")
    if backend is None:
        check("backend", False, f"Unknown backend '{backend_id}'.", hint="Pick a backend from the list.")
        backend = registry.resolve(None)
    else:
        check("backend", backend.info.available,
              f"{backend.info.name} is {'available' if backend.info.available else 'not available'}: "
              f"{backend.info.reason or 'ready'}.", hint="Install the runtime from the Environment page.")
    if backend and method not in backend.info.methods:
        check("method", False,
              f"{backend.info.name} cannot run the method '{method}'.",
              hint=f"Supported here: {', '.join(backend.info.methods)}.")

    paths = [str(path) for path in request.get("dataset_paths") or []]
    if not paths:
        check("dataset", False, "No dataset selected.",
              hint="Import a dataset, then choose it in the training configuration.")
    dataset_reports: list[dict[str, Any]] = []
    total_tokens = 0
    for path in paths:
        target = Path(path)
        if not target.exists():
            check(f"dataset:{target.name}", False, f"Dataset path missing: {target}",
                  hint="Re-import the dataset; the file or folder moved.")
            continue
        try:
            report = inspect_dataset(target, request.get("mapping") or {}, sample_size=400)
            dataset_reports.append({
                "path": str(target),
                "records": report["record_count"],
                "fields": report["field_names"],
                "duplicates": report["duplicates"],
                "average_length": report["length"]["average"],
            })
            estimate = report.get("token_estimate") or {}
            total_tokens += int(estimate.get("total") or 0)
            if report["record_count"] == 0:
                check(f"dataset:{target.name}", False, "Dataset is empty.", level="warning")
            elif report["empty_records"] and report["empty_records"] > report["record_count"] * 0.2:
                check(f"dataset:{target.name}", False,
                      f"{report['empty_records']} of {report['record_count']} records are empty.",
                      level="warning", hint="Run the cleaning tools before training.")
            else:
                check(f"dataset:{target.name}", True,
                      f"{report['record_count']:,} records, {report['size_human']}.")
        except ZxError as exc:
            check(f"dataset:{target.name}", False, exc.message, hint=exc.hint)

    model_path = request.get("base_model") or request.get("parent_checkpoint")
    model_report: dict[str, Any] | None = None
    if method in ("scratch",) and backend and backend.info.id == "tiny":
        check("model", True, "Training from scratch with the tiny backend: no base model required.")
    elif not model_path:
        check("model", False, "No base model selected.",
              hint="Import a model, or switch the method to training from scratch.")
    else:
        path = Path(str(model_path))
        if not path.exists():
            check("model", False, f"Model path missing: {path}",
                  hint="Re-import the model folder.")
        else:
            try:
                model_report = inspect_model(path)
                kind = (model_report.get("format") or {}).get("kind")
                params = (model_report.get("weights") or {}).get("parameter_count")
                check("model", True,
                      f"{path.name}: {kind}, {params:,} parameters" if params else f"{path.name}: {kind}")
                if kind == "gguf" and backend and backend.info.id == "hf":
                    check("model:trainable", False,
                          "GGUF weights are inference-only; training needs the original checkpoint.",
                          hint="Import the Transformers version of this model, or use GGUF for "
                               "serving/inference only.")
                if model_report.get("adapter_config") and method in ("full_finetune",):
                    check("model:adapter", False,
                          "This folder is a LoRA adapter; full fine-tuning needs the base model.",
                          level="warning")
                config = model_report.get("config") or {}
                if config.get("num_hidden_layers") and request.get("sequence_length"):
                    limit = config.get("max_position_embeddings")
                    if limit and int(request["sequence_length"]) > int(limit):
                        check("context", False,
                              f"Sequence length {request['sequence_length']} is above the model context "
                              f"({limit}).", level="warning")
                if backend and backend.info.id == "hf":
                    try:
                        from .hardware import module_available
                        if not module_available("transformers"):
                            check("runtime", False, "Transformers is not installed.",
                                  hint="Install the ML runtime from the Environment page.")
                    except Exception:  # pragma: no cover
                        pass
            except ZxError as exc:
                check("model", False, exc.message, hint=exc.hint)

    resume_from = request.get("resume_from")
    continuation: dict[str, Any] | None = None
    if resume_from or request.get("parent_checkpoint"):
        continuation = continuation_report(request, model_report)
        for item in continuation.get("checks", []):
            check(f"resume:{item['name']}", item["ok"], item["message"],
                  level=item.get("level", "warning"), hint=item.get("hint", ""))

    workspace = Path(request.get("workspace") or Path.home())
    output_dir = Path(request.get("output_dir") or (workspace / "jobs" / str(request.get("job_id") or "job") / "output"))
    try:
        ensure_dir(output_dir)
        probe = output_dir / ".zxtrain-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check("output", True, f"Output directory is writable: {output_dir}")
    except OSError as exc:
        check("output", False, f"Cannot write to {output_dir}: {exc}",
              hint="Choose another workspace folder in Settings.")

    plan = estimator.plan_run(request, hardware)
    if plan["disk"]["workspace_needed"]:
        free = next((disk["free"] for disk in hardware["disks"] if str(workspace).startswith(disk["path"])),
                    None) or (hardware["disks"][0]["free"] if hardware["disks"] else None)
        if free is not None and plan["disk"]["workspace_needed"] > free * 0.9:
            check("disk", False,
                  f"Estimated workspace need {human_bytes(plan['disk']['workspace_needed'])} exceeds free "
                  f"space ({human_bytes(free)}).",
                  hint="Lower the checkpoint limit or reduce the checkpoint frequency.")
        else:
            check("disk", True,
                  f"Free disk space {human_bytes(free) if free else 'unknown'} vs estimated "
                  f"{human_bytes(plan['disk']['workspace_needed'])}.")

    risk = plan["risk"]
    if risk in ("will_not_fit", "tight"):
        level = "error" if risk == "will_not_fit" else "warning"
        check("memory", False,
              f"Memory risk: {risk.replace('_', ' ')}. Estimated requirement "
              f"{human_bytes(plan['memory']['vram_estimate'])}.",
              level=level,
              hint="Reduce batch size, enable gradient checkpointing, use 4-bit loading, or shorten "
                   "the sequence length.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "plan": plan,
        "datasets": dataset_reports,
        "model": {
            "path": str(model_path) if model_path else None,
            "format": (model_report or {}).get("format", {}).get("kind"),
            "parameters": ((model_report or {}).get("weights") or {}).get("parameter_count"),
            "architecture": ((model_report or {}).get("architecture") or {}).get("model_type"),
        } if model_report else None,
        "continuation": continuation,
        "backend": backend.info.to_dict() if backend else None,
        "hardware_summary": {
            "gpus": [gpu["name"] for gpu in hardware["gpus"]],
            "ram": hardware["memory"]["total"],
            "cpu": hardware["cpu"]["model"],
        },
        "prepared_at": now_iso(),
    }


def continuation_report(request: dict[str, Any], model_report: dict[str, Any] | None) -> dict[str, Any]:
    """What exactly would be restored / changed when continuing a run."""
    resume_from = request.get("resume_from")
    parent = request.get("parent_checkpoint") or request.get("base_model")
    restored: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    verdict = "exact"

    if resume_from:
        directory = Path(str(resume_from))
        exists = directory.exists()
        checks.append({
            "name": "checkpoint",
            "ok": exists,
            "message": f"Checkpoint {directory.name} exists." if exists
            else f"Checkpoint folder is missing: {directory}",
            "level": "error",
            "hint": "Pick another checkpoint from the checkpoint list.",
        })
        state_files = {
            "model weights": ["model.safetensors", "pytorch_model.bin", "model.json", "adapter_model.safetensors"],
            "optimizer state": ["optimizer.pt", "optimizer.bin"],
            "scheduler state": ["scheduler.pt"],
            "rng state": ["rng_state.pth"],
            "trainer/training state": ["trainer_state.json", "model.json"],
        }
        for label, candidates in state_files.items():
            present = any((directory / name).exists() for name in candidates)
            restored.append({"item": label, "restored": present})
            if label != "model weights" and not present:
                verdict = "compatible" if verdict == "exact" else verdict
        if not all(item["restored"] for item in restored if item["item"] != "trainer/training state"):
            verdict = "compatible"
    else:
        restored.append({"item": "model weights", "restored": True})
        restored.append({"item": "optimizer state", "restored": False})
        restored.append({"item": "scheduler state", "restored": False})
        restored.append({"item": "rng state", "restored": False})
        verdict = "fresh_from_checkpoint" if False else "new_run"
        if parent:
            verdict = "new_run_from_model"

    parent_config = {}
    if parent and Path(str(parent)).exists():
        parent_config = read_json(Path(str(parent)) / "zxtrain-run.json", {}) or {}
        spec = parent_config.get("spec") or {}
        for field, label, compatibility in (
            ("learning_rate", "learning rate", "safe"),
            ("epochs", "epochs", "safe"),
            ("dataset_paths", "dataset", "safe"),
            ("lr_scheduler", "scheduler", "changes_optimizer"),
            ("gradient_accumulation", "gradient accumulation", "changes_optimizer"),
            ("precision", "precision", "may_affect_optimizer"),
        ):
            previous = spec.get(field)
            current = request.get(field)
            if previous is None:
                continue
            if current is not None and str(previous) != str(current):
                changed.append({"field": label, "from": previous, "to": current, "impact": compatibility})
                if compatibility != "safe" and verdict == "exact":
                    verdict = "compatible"
                elif compatibility != "safe":
                    verdict = "unsafe"

    if model_report:
        config = model_report.get("config") or {}
        if config.get("model_type"):
            restored.append({"item": "architecture", "restored": True,
                             "detail": config["model_type"]})
    return {
        "verdict": verdict,
        "verdict_text": {
            "exact": "Exact resume: the engine can restore weights, optimizer, scheduler and RNG state.",
            "compatible": "Compatible resume: training continues, but not every state file is available.",
            "unsafe": "Potentially unsafe resume: a changed setting may make the restored optimizer state "
                      "inconsistent. Training will still run, starting from the checkpoint's weights.",
            "new_run_from_model": "New run that continues from this model's weights, not from its "
                                  "optimizer state.",
            "new_run": "Fresh run.",
        }.get(verdict, verdict),
        "restored": restored,
        "changed": changed,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# Job execution
# --------------------------------------------------------------------------- #

class JobRunner:
    """Owns one job directory and the contract the UI reads."""

    def __init__(self, spec_path: Path):
        self.spec_path = Path(spec_path)
        self.request = read_json(self.spec_path, None)
        if not self.request:
            raise ZxError(
                code="job_spec_missing",
                message=f"Job spec is missing or unreadable: {self.spec_path}",
                hint="Re-create the job from the UI.",
            )
        self.job_dir = Path(self.request.get("job_dir") or self.spec_path.parent)
        ensure_dir(self.job_dir)
        ensure_dir(self.job_dir / "output")
        self.control = {"pause": False, "stop": False, "save_now": False}
        self._stop_watcher = threading.Event()
        self._events_path = self.job_dir / "events.jsonl"
        self._metrics_path = self.job_dir / "metrics.jsonl"
        self._status_path = self.job_dir / "status.json"
        self._control_path = self.job_dir / "control.json"
        self.started_at = time.time()
        self.status: dict[str, Any] = {
            "job_id": self.request.get("job_id") or self.job_dir.name,
            "kind": self.request.get("kind", "train"),
            "state": "running",
            "pid": os.getpid(),
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "heartbeat": time.time(),
            "step": 0,
            "total_steps": None,
            "loss": None,
            "message": "Starting",
            "backend": self.request.get("backend"),
            "method": self.request.get("method"),
            "job_dir": str(self.job_dir),
            "output_dir": str(self.job_dir / "output"),
            "checkpoints": [],
            "metrics": {},
            "result": None,
            "error": None,
        }

    # -- plumbing ---------------------------------------------------------- #
    def _watch_control(self) -> None:
        while not self._stop_watcher.is_set():
            payload = read_json(self._control_path, None)
            if isinstance(payload, dict):
                self.control.update({key: bool(value) for key, value in payload.items()})
            time.sleep(0.4)

    def start_control_watcher(self) -> None:
        thread = threading.Thread(target=self._watch_control, daemon=True)
        thread.start()

    def write_status(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at"] = now_iso()
        self.status["heartbeat"] = time.time()
        write_json_atomic(self._status_path, self.status)

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"ts": time.time(), "at": now_iso(), **event}
        append_jsonl(self._events_path, payload)
        if event.get("type") == "metrics":
            append_jsonl(self._metrics_path, payload)
            updates: dict[str, Any] = {"metrics": {**self.status.get("metrics", {}), **{
                key: value for key, value in event.items() if key not in ("type", "ts", "at")
            }}}
            for key in ("step", "total_steps", "epoch", "loss", "eval_loss", "learning_rate",
                        "tokens_per_second", "eta_seconds", "elapsed_seconds"):
                if event.get(key) is not None:
                    updates[key] = event[key]
            self.write_status(**updates)
        elif event.get("type") == "plan":
            self.write_status(total_steps=event.get("total_steps"),
                              message=f"Planned {event.get('total_steps')} optimizer steps",
                              plan=event)
        elif event.get("type") == "checkpoint":
            checkpoints = list(self.status.get("checkpoints") or [])
            checkpoints.append({
                "name": event.get("name"),
                "path": event.get("path"),
                "step": event.get("step"),
                "kind": event.get("kind"),
            })
            self.write_status(checkpoints=checkpoints, message=f"Checkpoint {event.get('name')} written")
        elif event.get("type") == "log":
            self.write_status(message=event.get("message"))
        elif event.get("type") == "done":
            self.write_status(message="Finished")
        line = json.dumps(payload, ensure_ascii=False)
        print(f"@@event {line}", flush=True)
        if event.get("type") == "log":
            print(f"[{event.get('level', 'info')}] {event.get('message')}", flush=True)

    # -- execution --------------------------------------------------------- #
    def run(self) -> dict[str, Any]:
        kind = str(self.request.get("kind") or "train")
        self.start_control_watcher()
        self.write_status(state="running", message="Pre-flight checks")
        self.emit({"type": "log", "level": "info",
                   "message": f"Job {self.status['job_id']} ({kind}) started on pid {os.getpid()}."})
        try:
            if kind == "train":
                result = self._run_train()
            elif kind == "evaluate":
                result = self._run_evaluate()
            elif kind == "generate":
                result = self._run_generate()
            elif kind == "merge":
                result = self._run_merge()
            elif kind == "quantize":
                result = self._run_quantize()
            elif kind == "tokenize":
                result = self._run_tokenize()
            elif kind == "server":
                result = self._run_server()
            else:
                raise ZxError(
                    code="unsupported_job",
                    message=f"Unknown job kind: {kind}",
                    hint="Supported kinds: train, evaluate, generate, merge, quantize, tokenize.",
                )
            self.write_status(state=result.get("status", "completed"), result=result,
                              message=result.get("status", "completed"))
            return result
        except ZxError as exc:
            self._fail(exc)
            return {"status": "failed", "error": exc.to_dict()}
        except KeyboardInterrupt:  # pragma: no cover - user interrupt
            self.write_status(state="cancelled", message="Cancelled")
            return {"status": "cancelled"}
        except BaseException as exc:  # noqa: BLE001 - everything must be recorded
            wrapped = from_exception(exc)
            self._fail(wrapped)
            return {"status": "failed", "error": wrapped.to_dict()}
        finally:
            self._stop_watcher.set()

    def _fail(self, error: ZxError) -> None:
        tail = tail_of(self._events_path, 60)
        self.emit({"type": "log", "level": "error", "message": error.message})
        self.write_status(state="failed", error=error.to_dict(), message=error.message, log_tail=tail)

    def _run_train(self) -> dict[str, Any]:
        request = self.request
        backend = registry.resolve(request.get("backend"))
        if backend is None:
            raise ZxError(code="backend_missing", message="No training backend is available.",
                          hint="Install the ML runtime from the Environment page.")
        if not backend.info.available:
            raise ZxError(code="backend_unavailable",
                          message=f"{backend.info.name} is not available: {backend.info.reason}",
                          hint="Install the required packages from the Environment page.")
        spec = build_spec(request, self.job_dir)
        self.emit({"type": "log", "level": "info", "message":
                   f"Backend: {backend.info.name}. Method: {spec.method}. Precision: {spec.precision}."
                   + (" Resuming from a checkpoint." if spec.resume_from else "")})
        result = backend.train(spec, self.emit, self.control)
        self._write_lineage(spec, result)
        return result

    def _run_evaluate(self) -> dict[str, Any]:
        backend = registry.resolve(self.request.get("backend"))
        if backend is None or not backend.info.available:
            raise ZxError(code="backend_unavailable", message="The selected backend is unavailable.",
                          hint="Install the ML runtime, or evaluate with the tiny backend.")
        return backend.evaluate(
            str(self.request.get("base_model") or self.request.get("parent_checkpoint")),
            [str(path) for path in self.request.get("dataset_paths") or []],
            self.request,
            self.emit,
        )

    def _run_generate(self) -> dict[str, Any]:
        backend = registry.resolve(self.request.get("backend"))
        if backend is None or not backend.info.available:
            raise ZxError(code="backend_unavailable", message="The selected backend is unavailable.",
                          hint="Load a tiny-backend model instead.")
        return backend.generate(
            str(self.request.get("base_model") or self.request.get("parent_checkpoint")),
            self.request,
            self.emit,
        )

    def _run_merge(self) -> dict[str, Any]:
        from .backends import hf

        if not hf.availability()[0]:
            raise ZxError(code="backend_unavailable", message="Merging adapters needs the PyTorch runtime.",
                          hint="Install the ML runtime from the Environment page.")
        return hf.merge_adapter(
            str(self.request.get("base_model")),
            self.request.get("merge_base"),
            str(self.request.get("output_dir")),
            self.emit,
        )

    def _run_quantize(self) -> dict[str, Any]:
        from .backends import hf

        if not hf.availability()[0]:
            raise ZxError(code="backend_unavailable",
                          message="Quantisation needs the PyTorch runtime (and bitsandbytes for int4/int8).",
                          hint="Install the ML runtime from the Environment page.")
        return hf.quantize_bnb(
            str(self.request.get("base_model")),
            str(self.request.get("output_dir")),
            int(self.request.get("bits") or 8),
            self.emit,
        )

    def _run_tokenize(self) -> dict[str, Any]:
        from .bpe import train_bpe, stats_for_texts
        from . import datasets as ds

        request = self.request
        paths = [str(path) for path in request.get("dataset_paths") or []]
        texts: list[str] = []
        for path in paths:
            texts.extend(ds.extract_texts(Path(path), request.get("mapping") or {}, limit=20_000,
                                          template=request.get("template", "chatml")))
        payload = train_bpe(texts, vocab_size=int(request.get("vocab_size") or 512),
                            progress=lambda step, total, frequency: self.emit({
                                "type": "metrics", "phase": "tokenize", "merges": step,
                                "total_merges": total, "last_merge_frequency": frequency,
                                "progress": round(step / max(1, total), 4),
                            }))
        from .bpe import BPETokenizer

        tokenizer = BPETokenizer(payload)
        output = Path(request.get("output_dir") or (self.job_dir / "output")) / "tokenizer.json"
        ensure_dir(output.parent)
        tokenizer.save(output)
        stats = stats_for_texts(tokenizer, texts[:2000])
        result = {"status": "completed", "backend": "bpe", "tokenizer": str(output),
                  "vocab_size": tokenizer.vocab_size, "stats": stats, "texts": len(texts)}
        self.emit({"type": "done", "result": result})
        return result

    def _run_server(self) -> dict[str, Any]:
        """Serve a model over the local HTTP API until the UI asks it to stop."""
        from . import server as server_module

        request = self.request
        backend = registry.resolve(request.get("backend"))
        if backend is None or not backend.info.available:
            raise ZxError(
                code="backend_unavailable",
                message="The selected inference backend is not available.",
                hint="Load a tiny-backend model, or install the ML runtime from the Environment page.",
            )
        model = str(request.get("base_model") or request.get("parent_checkpoint") or "")
        if not model or not Path(model).exists():
            raise ZxError(
                code="model_missing",
                message=f"Model path does not exist: {model or '(empty)'}",
                hint="Pick a model from the library in the Deploy panel.",
            )
        instance = server_module.InferenceServer(
            model_path=model,
            backend=backend,
            host=str(request.get("host") or "127.0.0.1"),
            port=int(request.get("port") or 8080),
            log_file=self.job_dir / "requests.jsonl",
            emit=self.emit,
            concurrency=int(request.get("concurrency") or 2),
        )
        result = instance.serve_until(lambda: bool(self.control.get("stop")))
        result["status"] = "completed"
        self.emit({"type": "done", "result": result})
        return result

    def _write_lineage(self, spec: TrainSpec, result: dict[str, Any]) -> None:
        parent = {
            "kind": "checkpoint" if spec.resume_from else "model",
            "path": str(spec.resume_from or spec.parent_checkpoint or spec.base_model or ""),
            "job_id": self.request.get("parent_job_id"),
        }
        lineage = {
            "job_id": spec.job_id,
            "created_at": now_iso(),
            "parent": parent if parent["path"] else None,
            "child": {
                "model_dir": result.get("model_dir"),
                "method": spec.method,
                "backend": (result.get("backend") or self.request.get("backend")),
                "steps": result.get("steps"),
            },
            "dataset_paths": spec.dataset_paths,
            "resumed": bool(spec.resume_from),
            "restored": {
                "weights": True,
                "optimizer": bool(spec.resume_from and (Path(str(spec.resume_from)) / "optimizer.pt").exists())
                or bool(spec.resume_from and (Path(str(spec.resume_from)) / "model.json").exists()),
                "rng": bool(spec.resume_from and (Path(str(spec.resume_from)) / "rng_state.pth").exists())
                or bool(spec.resume_from and (Path(str(spec.resume_from)) / "model.json").exists()),
            },
            "note": result.get("lineage_note", ""),
        }
        write_json_atomic(self.job_dir / "lineage.json", lineage)
        append_jsonl(self.job_dir.parent / "lineage.jsonl", lineage)


# --------------------------------------------------------------------------- #
# Status helpers used by the UI and by recovery
# --------------------------------------------------------------------------- #

def tail_of(path: Path, lines: int = 40) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def read_status(job_dir: Path) -> dict[str, Any]:
    payload = read_json(Path(job_dir) / "status.json", None)
    if not payload:
        return {"state": "unknown", "message": "No status file yet."}
    return payload


def process_alive(pid: Any) -> bool:
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def reconcile(jobs_root: Path) -> list[dict[str, Any]]:
    """Reconnect to or bury jobs whose process is no longer running."""
    jobs_root = Path(jobs_root)
    if not jobs_root.exists():
        return []
    findings: list[dict[str, Any]] = []
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir():
            continue
        status = read_json(job_dir / "status.json", None)
        if not status:
            continue
        state = status.get("state")
        if state not in ("running", "queued", "paused"):
            continue
        alive = process_alive(status.get("pid"))
        if alive:
            findings.append({"job_id": job_dir.name, "state": "running", "action": "reconnected",
                             "message": "Training process is still alive; the UI reattached to its logs."})
            continue
        heartbeat_age = time.time() - float(status.get("heartbeat") or 0)
        checkpoints = status.get("checkpoints") or []
        resumable = None
        for checkpoint in reversed(checkpoints):
            path = Path(str(checkpoint.get("path") or ""))
            if path.exists():
                resumable = str(path)
                break
        status.update({
            "state": "interrupted",
            "message": "The training process is no longer running.",
            "interrupted_at": now_iso(),
            "heartbeat_age_seconds": round(heartbeat_age, 1),
            "resumable_from": resumable,
            "log_tail": tail_of(job_dir / "events.jsonl", 40),
        })
        write_json_atomic(job_dir / "status.json", status)
        findings.append({
            "job_id": job_dir.name,
            "state": "interrupted",
            "action": "marked_interrupted",
            "resumable_from": resumable,
            "message": ("The process died. Checkpoints are intact — you can resume from the last one."
                        if resumable else "The process died before the first checkpoint was written."),
        })
    return findings


def job_summary(job_dir: Path) -> dict[str, Any]:
    status = read_status(job_dir)
    metrics = []
    metrics_path = Path(job_dir) / "metrics.jsonl"
    if metrics_path.exists():
        try:
            for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            metrics = []
    spec = read_json(Path(job_dir) / "spec.json", {}) or {}
    return {
        "job_id": status.get("job_id") or Path(job_dir).name,
        "job_dir": str(job_dir),
        "state": status.get("state", "unknown"),
        "kind": status.get("kind") or spec.get("kind"),
        "backend": status.get("backend") or spec.get("backend"),
        "method": status.get("method") or spec.get("method"),
        "step": status.get("step"),
        "total_steps": status.get("total_steps"),
        "loss": status.get("loss"),
        "message": status.get("message"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "checkpoints": status.get("checkpoints") or [],
        "result": status.get("result"),
        "error": status.get("error"),
        "model": spec.get("base_model"),
        "datasets": spec.get("dataset_paths") or [],
        "metrics": metrics,
        "config": spec,
    }


def format_of(path: str | Path) -> str:
    try:
        return str(detect_format(Path(path)).get("kind"))
    except Exception:  # pragma: no cover - defensive
        return "unknown"
