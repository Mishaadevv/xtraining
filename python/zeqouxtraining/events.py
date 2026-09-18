"""Event protocol between the Python backend and the ZeqouXTraining shell.

Wire format: one JSON object per line on stdout.

    {"event": "<name>", "detail": { ... }}

``event`` names used across the app (kept in one place so the renderer and the
backend cannot drift apart):

    ready                backend booted and is handling a command
    stage                coarse progress step of a long task
    log                  human-readable log line (level: info|warn|error)
    hardware             detected hardware snapshot
    dataset-progress     dataset loading progress
    dataset-report       dataset validation report
    model-info           resolved model metadata
    training-status      phase change of a training run
    training-progress    per-step metrics
    training-checkpoint  a checkpoint was written
    training-complete    run finished successfully
    training-error       run failed; carries message + hint + traceback
    inference-token      one streamed token of generation
    inference-result     final generation result
    result               terminal payload of a one-shot command
    error                terminal error of a one-shot command
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# The real stdout, captured before anything is allowed to replace it.
_EVENT_STREAM = sys.stdout


def install_diagnostics_redirect() -> None:
    """Send everything that is not an event to stderr.

    Called by long-running commands (training, inference) after the process has
    started. ``emit`` keeps writing to the original stdout handle, so the
    protocol survives regardless of what third-party code prints.
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass

    # Third-party libraries go to stderr from here on.
    sys.stdout = sys.stderr

    # Keep Hugging Face / tqdm from writing progress bars anywhere.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")


def emit(event: str, detail: dict[str, Any] | None = None) -> None:
    """Write one protocol event.

    Never raises into the caller: a broken stdout pipe (the app was closed)
    should not turn into a confusing training crash.
    """
    payload = {"event": event, "detail": detail or {}, "ts": time.time()}
    try:
        _EVENT_STREAM.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        _EVENT_STREAM.flush()
    except Exception:
        pass


def log(message: str, level: str = "info") -> None:
    """Human-readable line for the technical log panel (goes to stderr)."""
    emit("log", {"level": level, "message": message})


def stage(name: str, message: str, **extra: Any) -> None:
    emit("stage", {"stage": name, "message": message, **extra})


def result(payload: dict[str, Any]) -> None:
    emit("result", payload)


def fail(message: str, *, hint: str = "", code: str = "error",
         traceback_text: str = "", extra: dict[str, Any] | None = None) -> None:
    detail = {"message": message, "hint": hint, "code": code}
    if traceback_text:
        detail["traceback"] = traceback_text
    if extra:
        detail.update(extra)
    emit("error", detail)
