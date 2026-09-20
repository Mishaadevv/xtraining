"""Training job runner.

Owns everything that is not backend-specific: reading the job file, resolving
the dataset into training samples, selecting a backend, enforcing the
stop/pause files, and turning failures into structured, readable events.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from . import events
from .backends.base import RunContext
from .backends.registry import default_backend_name, get_backend
from .config import normalize
from .datasets import DatasetError, detect_mapping, load_hf_dataset, load_records, normalize as normalize_samples
from .errors import humanize
from .models import ModelError, inspect as inspect_model

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STOPPED = 130


def read_job(job_path: str | Path) -> dict[str, Any]:
    path = Path(job_path)
    if not path.is_file():
        raise FileNotFoundError(f"Job file not found: {path}")
    raw = path.read_text(encoding="utf-8-sig")
    return json.loads(raw)


def _resolve_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and normalise the dataset described by the config."""
    dataset = config.get("dataset") or {}
    mapping = dataset.get("mapping")
    started = time.time()

    if dataset.get("hf_id"):
        events.stage("dataset", f"Downloading Hugging Face dataset {dataset['hf_id']}")
        records = load_hf_dataset(str(dataset["hf_id"]), split=str(dataset.get("split") or "train"))
        source = {"path": None, "name": str(dataset["hf_id"]), "format": "hf", "records": len(records)}
    else:
        path = dataset.get("path")
        if not path:
            raise DatasetError("No dataset path or Hugging Face dataset id was provided.", code="no_dataset")

        def progress(message: str, index: int, total: int) -> None:
            events.emit("dataset-progress", {"message": message, "index": index, "total": total})

        events.stage("dataset", f"Loading {Path(str(path)).name}")
        records, meta = load_records(str(path), fmt=str(dataset.get("format") or "auto"), progress=progress)
        source = {
            "path": meta["path"], "name": meta["name"],
            "format": meta["format"], "records": meta["records"],
        }

    resolved_mapping = mapping or detect_mapping(records)
    samples = normalize_samples(records, resolved_mapping)

    events.emit("dataset-report", {
        "source": source,
        "mapping": resolved_mapping,
        "records": len(records),
        "samples": len(samples),
        "load_seconds": round(time.time() - started, 2),
    })
    return samples, source


def run_job(job_path: str | Path) -> int:
    """Execute one training job. Returns the process exit code."""
    events.emit("ready", {"command": "train", "job": str(job_path)})

    try:
        job = read_job(job_path)
    except Exception as exc:
        events.fail(f"Could not read the job file: {exc}",
                    hint="Create the run again from the New Training screen.", code="bad_job")
        return EXIT_ERROR

    config = normalize(job.get("config") or {})
    run_dir = Path(job.get("run_dir") or ".")
    stop_file = Path(job.get("stop_file") or (run_dir / ".stop"))
    pause_file = Path(job.get("pause_file") or (run_dir / ".pause"))
    run_dir.mkdir(parents=True, exist_ok=True)

    def stop_requested() -> bool:
        return stop_file.exists()

    def pause_requested() -> bool:
        return pause_file.exists()

    events.emit("training-status", {
        "message": "Preparing run", "phase": "prepare",
        "run_dir": str(run_dir.resolve()), "method": config["method"],
    })

    started = time.time()
    try:
        samples, source = _resolve_samples(config)
    except DatasetError as exc:
        events.fail(exc.message, hint=exc.hint, code=exc.code)
        return EXIT_ERROR
    except Exception as exc:
        info = humanize(exc, stage="dataset")
        events.fail(info["message"], hint=info["hint"], code=info["code"],
                    traceback_text=info["traceback"])
        return EXIT_ERROR

    if not samples:
        events.fail(
            "The dataset produced no training samples.",
            hint="Open the Check step and resolve the reported dataset issues, then run again.",
            code="no_samples",
        )
        return EXIT_ERROR

    try:
        # From-scratch runs have no base model to inspect; the backend builds
        # the architecture from the config instead.
        if config.get("method") == "scratch":
            model_info = job.get("model_info") or {}
        else:
            model_info = job.get("model_info") or inspect_model(str(config["base_model"]))
    except ModelError as exc:
        events.fail(exc.message, hint=exc.hint, code=exc.code)
        return EXIT_ERROR
    except Exception as exc:
        info = humanize(exc, stage="model")
        events.fail(info["message"], hint=info["hint"], code=info["code"],
                    traceback_text=info["traceback"])
        return EXIT_ERROR

    # The scratch method is implemented by its own backend; route it there
    # unless the job file already pinned one explicitly.
    backend_name = job.get("backend") or (
        "scratch" if config.get("method") == "scratch" else default_backend_name()
    )
    backend_cls = get_backend(backend_name)
    if backend_cls is None:
        events.fail(
            f"Unknown training backend '{backend_name}'.",
            hint="Pick a backend that exists in this build.",
            code="unknown_backend",
        )
        return EXIT_ERROR

    backend = backend_cls()
    context = RunContext(
        job_id=str(job.get("job_id") or run_dir.name),
        config=config,
        samples=samples,
        run_dir=run_dir,
        model_info=model_info,
        hardware=job.get("hardware") or {},
        emit=events.emit,
        stop_requested=stop_requested,
        pause_requested=pause_requested,
        total_samples=len(samples),
    )

    preflight = backend.preflight(context)
    blocking = [issue for issue in preflight if issue.get("severity") == "error"]
    if blocking:
        events.fail(
            blocking[0]["message"],
            hint=blocking[0].get("hint", ""),
            code=blocking[0].get("code", "preflight"),
            extra={"issues": preflight},
        )
        return EXIT_ERROR
    for issue in preflight:
        events.emit("training-status", {
            "message": issue["message"], "phase": "warning", "severity": issue.get("severity"),
        })

    # Hand the resolved dataset size to the backend so metadata is complete.
    config["dataset_records"] = source.get("records")
    config["dataset_usable"] = len(samples)

    try:
        result = backend.run(context)
    except KeyboardInterrupt:
        events.emit("training-complete", {
            "status": "stopped", "message": "Training stopped.", "stopped": True,
            "output_dir": str(run_dir.resolve()),
        })
        return EXIT_STOPPED
    except Exception as exc:
        info = humanize(exc, stage="training")
        events.fail(info["message"], hint=info["hint"], code=info["code"],
                    traceback_text=info["traceback"])
        return EXIT_ERROR

    summary = {
        "job_id": context.job_id,
        "status": result.status,
        "output_dir": result.output_dir,
        "final_loss": result.final_loss,
        "total_steps": result.total_steps,
        "last_checkpoint": result.last_checkpoint,
        "train_mode": config["method"],
        "duration_seconds": int(time.time() - started),
        "history": result.history,
        "metrics": result.metrics,
        "dataset": source,
    }
    try:
        (run_dir / "run.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        events.log(f"Could not write run.json: {exc}", level="warn")

    messages = {
        "completed": "Training completed successfully.",
        "paused": "Training paused and checkpoints saved. Resume from the last checkpoint at any time.",
        "stopped": "Training stopped. Checkpoints were saved.",
    }
    events.emit("training-complete", {
        "status": result.status,
        "message": messages.get(result.status, "Training finished."),
        "output_dir": result.output_dir,
        "final_loss": result.final_loss,
        "total_steps": result.total_steps,
        "last_checkpoint": result.last_checkpoint,
        "train_mode": config["method"],
        "stopped": result.status in ("stopped", "paused"),
        "paused": result.status == "paused",
        "duration_seconds": int(time.time() - started),
        "history": result.history,
        "metrics": result.metrics,
        "dataset": source,
    })
    return EXIT_OK


def last_exit_code_for_exception(exc: BaseException) -> int:
    events.fail(str(exc), traceback_text=traceback.format_exc())
    return EXIT_ERROR
