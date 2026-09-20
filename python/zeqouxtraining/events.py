"""Wire protocol between the backend and the desktop shell.

One JSON object per line on the *original* stdout:

    {"event": "<name>", "detail": {...}, "ts": <unix time>}

Event names (single source of truth for both sides):

    ready                 the backend booted and is handling a command
    stage                 coarse step of a long task
    log                   human-readable log line (level: info|warn|error)
    dataset-progress      dataset loading progress
    dataset-report        summary of what the run loaded
    model-info            resolved model metadata
    training-status       phase change of a training run
    training-progress     per-step metrics
    training-checkpoint   a checkpoint was written
    training-complete     the run finished (completed | stopped | paused)
    inference-token       one streamed token of generation
    inference-result      final generation result
    inference-error       generation failed
    result                terminal payload of a one-shot command
    error                 terminal error of a one-shot command
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# The real stdout, captured before anything is allowed to replace it.
_EVENT_STREAM = sys.stdout

# Emitted by the trainer and forwarded verbatim to the renderer.
TRAINING_EVENTS = (
    "training-status",
    "training-progress",
    "training-checkpoint",
    "training-complete",
    "training-error",
)


def install_diagnostics_redirect() -> None:
    """Send everything that is not an event to stderr.

    Long-running commands (train, infer) call this after booting. ``emit``
    keeps writing to the captured stdout handle, so the protocol survives
    whatever third-party libraries print to ``sys.stdout`` afterwards.
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass

    sys.stdout = sys.stderr

    # Keep Hugging Face / tqdm from drawing progress bars anywhere.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")


def emit(event: str, detail: dict[str, Any] | None = None) -> None:
    """Write one protocol event. Never raises: a closed pipe must not crash a run."""
    payload = {"event": event, "detail": detail or {}, "ts": time.time()}
    try:
        _EVENT_STREAM.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        _EVENT_STREAM.flush()
    except Exception:
        pass


def log(message: str, level: str = "info") -> None:
    """Human-readable line for the technical log panel."""
    emit("log", {"level": level, "message": message})


def stage(name: str, message: str, **extra: Any) -> None:
    emit("stage", {"stage": name, "message": message, **extra})


def result(payload: dict[str, Any]) -> None:
    emit("result", payload)


def fail(message: str, *, hint: str = "", code: str = "error",
         traceback_text: str = "", extra: dict[str, Any] | None = None) -> None:
    detail: dict[str, Any] = {"message": message, "hint": hint, "code": code}
    if traceback_text:
        detail["traceback"] = traceback_text
    if extra:
        detail.update(extra)
    emit("error", detail)
