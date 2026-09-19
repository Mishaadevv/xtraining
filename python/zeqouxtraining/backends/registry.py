"""Backend registry.

Register a new backend here and it appears in the UI automatically, including
its availability state and the exact packages it is missing.
"""

from __future__ import annotations

from typing import Any

from .base import TrainingBackend

_REGISTERED: dict[str, type[TrainingBackend]] = {}


def register(backend: type[TrainingBackend]) -> type[TrainingBackend]:
    _REGISTERED[backend.name] = backend
    return backend


def _load_builtin() -> None:
    if "hf-peft" in _REGISTERED:
        return
    from .hf_peft import HuggingFacePeftBackend  # noqa: PLC0415
    from .scratch import ScratchBackend  # noqa: PLC0415

    register(HuggingFacePeftBackend)
    register(ScratchBackend)


def all_backends() -> dict[str, type[TrainingBackend]]:
    _load_builtin()
    return dict(_REGISTERED)


def get_backend(name: str) -> type[TrainingBackend] | None:
    _load_builtin()
    return _REGISTERED.get(name)


def default_backend_name() -> str:
    return "hf-peft"


def available_backends() -> list[str]:
    return [name for name, backend in all_backends().items() if backend.is_available()]


def backend_capabilities() -> list[dict[str, Any]]:
    _load_builtin()
    return [backend.capabilities() for backend in _REGISTERED.values()]
