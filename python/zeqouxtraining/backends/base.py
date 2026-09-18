"""Training backend interface.

The training engine is intentionally not tied to one framework. A backend
declares which methods it can run, reports its own capabilities, and returns a
uniform result. Adding a backend later (Unsloth, Axolotl, TRL, llama.cpp, an
ONNX runtime, a remote cluster) means adding one module and registering it —
no changes to the runner, the event protocol or the UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

EmitFn = Callable[[str, dict[str, Any]], None]
FlagFn = Callable[[], bool]


@dataclass
class RunContext:
    """Everything a backend needs for one training run."""

    job_id: str
    config: dict[str, Any]
    samples: list[dict[str, Any]]
    run_dir: Path
    model_info: dict[str, Any]
    hardware: dict[str, Any]
    emit: EmitFn
    stop_requested: FlagFn
    pause_requested: FlagFn
    total_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def requested_stop(self) -> bool:
        return self.stop_requested()

    def requested_pause(self) -> bool:
        return self.pause_requested()


@dataclass
class RunResult:
    status: str  # completed | stopped | paused
    final_loss: float | None = None
    total_steps: int = 0
    history: dict[str, list[Any]] = field(default_factory=dict)
    output_dir: str = ""
    last_checkpoint: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class TrainingBackend(ABC):
    """Base class for every training backend."""

    #: Stable identifier used in the job file and by the UI.
    name: str = "base"
    #: Human-readable label.
    label: str = "Base"
    #: Training methods this backend implements.
    methods: tuple[str, ...] = ()
    #: Importable modules this backend needs.
    requires: tuple[str, ...] = ()

    # ---------------------------------------------------------------- status
    @classmethod
    def missing_requirements(cls) -> list[str]:
        import importlib.util

        missing: list[str] = []
        for module in cls.requires:
            try:
                if importlib.util.find_spec(module) is None:
                    missing.append(module)
            except (ImportError, ValueError):
                missing.append(module)
        return missing

    @classmethod
    def is_available(cls) -> bool:
        return not cls.missing_requirements()

    @classmethod
    def capabilities(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "label": cls.label,
            "methods": list(cls.methods),
            "requires": list(cls.requires),
            "available": cls.is_available(),
            "missing": cls.missing_requirements(),
        }

    # -------------------------------------------------------------- preflight
    def preflight(self, ctx: RunContext) -> list[dict[str, Any]]:
        """Problems that would make this run fail. Empty list means "go"."""
        issues: list[dict[str, Any]] = []
        if not self.is_available():
            issues.append({
                "severity": "error",
                "code": "backend_unavailable",
                "message": f"{self.label} needs: {', '.join(self.missing_requirements())}.",
                "hint": "Install the ML runtime in Settings → Environment.",
            })
        if ctx.config.get("method") not in self.methods:
            issues.append({
                "severity": "error",
                "code": "unsupported_method",
                "message": f"{self.label} does not implement '{ctx.config.get('method')}'.",
                "hint": f"Supported methods: {', '.join(self.methods)}.",
            })
        if not ctx.samples:
            issues.append({
                "severity": "error",
                "code": "no_samples",
                "message": "The dataset produced no training samples.",
                "hint": "Fix the dataset issues reported in the Check step.",
            })
        return issues

    # ------------------------------------------------------------------ run
    @abstractmethod
    def run(self, ctx: RunContext) -> RunResult:
        """Execute the run. Must emit training-* events as it goes."""
        raise NotImplementedError
