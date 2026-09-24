"""Structured errors and diagnostics.

Every failure raised by the engine carries a machine readable code, a human
summary and an actionable hint. The UI never has to invent an explanation for a
failure, and never has to show "something went wrong".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ZxError(Exception):
    """A failure the engine understands well enough to explain."""

    code: str
    message: str
    hint: str = ""
    detail: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "detail": self.detail,
            "context": self.context,
        }


# Codewords used by the UI to decide which panel to open.
DIAGNOSIS_PATTERNS: list[tuple[str, str, str]] = [
    (
        "out_of_memory",
        "cuda out of memory",
        "Reduce per-device batch size, raise gradient accumulation to keep the effective "
        "batch size, enable gradient checkpointing, or load the model in 4-bit.",
    ),
    (
        "out_of_memory",
        "out of memory",
        "Lower the batch size and sequence length first; those two dominate activation memory.",
    ),
    (
        "disk_full",
        "no space left",
        "Free disk space in the workspace, or point checkpoints at another drive in Settings.",
    ),
    (
        "missing_dependency",
        "no module named",
        "The Python runtime is missing a package the selected backend needs. Open "
        "Environment to install the ML runtime into an isolated environment.",
    ),
    (
        "driver_mismatch",
        "cuda driver version is insufficient",
        "The installed PyTorch build expects a newer NVIDIA driver. Install a PyTorch "
        "build matching the driver, or run on CPU.",
    ),
    (
        "unsupported_architecture",
        "cannot import",
        "This model needs custom code (trust_remote_code). Enable custom architectures "
        "for this model only, after reviewing the source.",
    ),
    (
        "corrupt_checkpoint",
        "error while deserializing header",
        "The checkpoint file is truncated or damaged. Pick another checkpoint, or re-run "
        "training from the last healthy one.",
    ),
    (
        "tokenizer_error",
        "sentencepiece",
        "The tokenizer needs an extra package (sentencepiece or tiktoken). Install it from "
        "the Environment page.",
    ),
    (
        "permission_denied",
        "permission denied",
        "The workspace folder is not writable. Choose another workspace in Settings.",
    ),
    (
        "network_blocked",
        "connection error",
        "The Hub is unreachable. Switch to Offline mode in Settings to work with local "
        "files only.",
    ),
    (
        "dataset_invalid",
        "unicode decode error",
        "A dataset file is not valid UTF-8. Re-import it with an explicit encoding, or "
        "quarantine the bad records from the Cleaning tab.",
    ),
]


def diagnose(raw: str) -> tuple[str, str, str]:
    """Map raw process output onto (code, plain-language cause, recommendation)."""
    lowered = raw.lower()
    for code, needle, advice in DIAGNOSIS_PATTERNS:
        if needle in lowered:
            return code, needle, advice
    return "unknown", "", "Open the job log for the raw output; the engine could not classify this failure."


def from_exception(exc: BaseException, code: str = "engine_error") -> ZxError:
    """Wrap an arbitrary exception in a ZxError without losing the traceback tail."""
    if isinstance(exc, ZxError):
        return exc
    import traceback

    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
    diag_code, needle, advice = diagnose(f"{type(exc).__name__}: {exc}")
    return ZxError(
        code=diag_code if diag_code != "unknown" else code,
        message=f"{type(exc).__name__}: {exc}",
        hint=advice,
        detail=detail,
        context={"matched": needle} if needle else {},
    )


def require(condition: bool, code: str, message: str, hint: str = "") -> None:
    if not condition:
        raise ZxError(code=code, message=message, hint=hint)
