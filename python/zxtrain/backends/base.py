"""Backend interface shared by every training engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class BackendInfo:
    """What a backend is and whether it can run here."""

    id: str
    name: str
    description: str
    available: bool
    reason: str = ""
    requires: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    supports_gpu: bool = False
    supports_resume: bool = True
    supports_pause: bool = False
    supports_streaming_inference: bool = True
    supports_continue: bool = True
    supports_from_scratch: bool = False
    supports_adapters: bool = False
    supports_preference_training: bool = False
    speed_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "available": self.available,
            "reason": self.reason,
            "requires": self.requires,
            "methods": self.methods,
            "supports_gpu": self.supports_gpu,
            "supports_resume": self.supports_resume,
            "supports_pause": self.supports_pause,
            "supports_streaming_inference": self.supports_streaming_inference,
            "supports_continue": self.supports_continue,
            "supports_from_scratch": self.supports_from_scratch,
            "supports_adapters": self.supports_adapters,
            "supports_preference_training": self.supports_preference_training,
            "speed_note": self.speed_note,
        }


@dataclass
class TrainSpec:
    """Everything the backend needs for one run, resolved by the engine."""

    job_id: str
    output_dir: Path
    method: str = "lora"
    base_model: str | None = None
    parent_checkpoint: str | None = None
    resume_from: str | None = None
    dataset_paths: list[str] = field(default_factory=list)
    dataset_weights: list[float] = field(default_factory=list)
    mapping: dict[str, Any] = field(default_factory=dict)
    template: str = "chatml"
    tokenizer_path: str | None = None
    sequence_length: int = 256
    batch_size: int = 4
    gradient_accumulation: int = 1
    epochs: float = 1.0
    max_steps: int = 0
    learning_rate: float = 3e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    max_grad_norm: float = 1.0
    precision: str = "fp32"
    quantization: str = "none"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    gradient_checkpointing: bool = False
    seed: int = 42
    eval_every: int = 50
    save_every: int = 100
    logging_every: int = 10
    checkpoint_limit: int = 3
    device: str = "auto"
    eval_ratio: float = 0.05
    hidden_size: int = 32
    context_length: int = 8
    vocab_size: int = 512
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in self.__dict__.items()
        }
        return payload


class Backend(Protocol):
    info: BackendInfo

    def train(self, spec: TrainSpec, progress: ProgressFn, control: dict[str, Any]) -> dict[str, Any]:
        """Run training; emit progress events; return a result summary."""

    def generate(self, model_path: str, request: dict[str, Any], emit: ProgressFn) -> dict[str, Any]:
        """Generate text, streaming partial output through `emit`."""

    def evaluate(self, model_path: str, dataset_paths: list[str], request: dict[str, Any],
                 progress: ProgressFn) -> dict[str, Any]:
        """Measure real quality metrics on real data."""
