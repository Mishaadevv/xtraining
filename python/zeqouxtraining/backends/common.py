"""Shared helpers for the built-in training backends.

Small pure functions used by both the HF+PEFT and scratch backends, kept in
one place so their behaviour stays identical.
"""

from __future__ import annotations

import inspect
import json
from typing import Any


def supported_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments the installed library version does not accept.

    ``TrainingArguments`` field names changed across transformers versions;
    filtering against the real signature beats guessing.
    """
    try:
        accepted = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def load_kwargs(loader: Any, base: dict[str, Any]) -> dict[str, Any]:
    """Handle the torch_dtype → dtype rename across transformers versions."""
    try:
        accepted = set(inspect.signature(loader).parameters)
    except (TypeError, ValueError):
        return base
    kwargs = dict(base)
    if "dtype" in accepted and "dtype" not in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype", None)
    elif "torch_dtype" not in accepted:
        kwargs.pop("torch_dtype", None)
    return {key: value for key, value in kwargs.items() if key in accepted and value is not None}


def jsonable(value: Any) -> Any:
    """Convert a config tree into something JSON can serialise."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump_json(path: Any, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=True, indent=2), encoding="utf-8")
