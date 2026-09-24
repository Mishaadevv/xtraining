"""Backend registry — one place that answers 'what can run here?'."""

from __future__ import annotations

from typing import Any

from ..hardware import module_available
from . import hf, tiny


def all_backends() -> list[Any]:
    backends: list[Any] = [tiny.TinyBackend(), hf.HFBackend()]
    try:  # TRL adds preference optimisation on top of the Transformers backend.
        if module_available("trl"):
            backends[1].info.supports_preference_training = True
            if "preference" not in backends[1].info.methods:
                backends[1].info.methods.append("preference")
    except Exception:  # pragma: no cover - defensive
        pass
    return backends


def describe() -> list[dict[str, Any]]:
    return [backend.info.to_dict() for backend in all_backends()]


def get(identifier: str):
    for backend in all_backends():
        if backend.info.id == identifier:
            return backend
    return None


def resolve(identifier: str | None) -> Any:
    """Pick the requested backend, or the best available one."""
    if identifier:
        backend = get(identifier)
        if backend is None:
            return None
        return backend
    for backend in all_backends():
        if backend.info.available and backend.info.id == "hf":
            return backend
    return all_backends()[0]


def supported_methods() -> list[dict[str, Any]]:
    """Method → backends that can really run it here."""
    catalogue: dict[str, dict[str, Any]] = {}

    def register(method: str, backend, available: bool, reason: str) -> None:
        entry = catalogue.setdefault(method, {"method": method, "backends": [], "available": False,
                                              "reasons": []})
        entry["backends"].append(backend.info.id)
        if available:
            entry["available"] = True
        elif reason:
            entry["reasons"].append(f"{backend.info.name}: {reason}")

    for backend in all_backends():
        for method in backend.info.methods:
            register(method, backend, backend.info.available,
                     "" if backend.info.available else backend.info.reason)
    # Methods that exist in the product but need a runtime that is missing here.
    for method, note in (
        ("preference", "Preference optimisation (DPO/ORPO/GRPO) requires the TRL runtime."),
        ("grpo", "GRPO requires the TRL runtime with a reward setup."),
        ("orpo", "ORPO requires the TRL runtime."),
    ):
        entry = catalogue.setdefault(method, {"method": method, "backends": [], "available": False,
                                              "reasons": []})
        if not module_available("trl"):
            entry["reasons"].append(note)
            entry["available"] = entry["available"] and False
    return sorted(catalogue.values(), key=lambda item: item["method"])
