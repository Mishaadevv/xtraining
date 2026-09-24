"""Adapter (LoRA/PEFT) discovery and merge planning.

An adapter is just a folder with ``adapter_config.json`` next to its weight
files. Nothing here guesses: the base model, rank, alpha and target modules are
read from that config, and a merge plan is refused when the adapter does not
name a base model or the base model is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import models
from .errors import ZxError
from .util import human_bytes


def is_adapter(path: str | Path) -> bool:
    root = Path(path)
    return (root / "adapter_config.json").is_file()


def describe(path: str | Path) -> dict[str, Any]:
    """Everything the Adapters page needs about one adapter folder."""
    root = Path(path).expanduser()
    if not root.exists():
        raise ZxError(code="adapter_missing", message=f"Adapter folder does not exist: {root}")
    if not is_adapter(root):
        raise ZxError(
            code="not_an_adapter",
            message=f"{root.name} has no adapter_config.json.",
            hint="Point the Adapters page at a PEFT/LoRA output folder.",
        )
    report = models.inspect(str(root))
    adapter = report.get("adapter_details") or {}
    files = models.list_weight_files(root)
    weight_files = [*files.get("safetensors", []), *files.get("bin", [])]
    total = 0
    for item in weight_files:
        try:
            total += Path(item).stat().st_size
        except OSError:
            continue
    return {
        "path": str(root),
        "name": root.name,
        "base_model": adapter.get("base_model"),
        "base_model_local": bool(adapter.get("base_model") and Path(str(adapter["base_model"])).exists()),
        "peft_type": adapter.get("peft_type"),
        "task_type": adapter.get("task_type"),
        "rank": adapter.get("r"),
        "alpha": adapter.get("lora_alpha"),
        "dropout": adapter.get("lora_dropout"),
        "scaling": adapter.get("scaling"),
        "target_modules": adapter.get("target_modules") or [],
        "modules_to_save": adapter.get("modules_to_save"),
        "bias": adapter.get("bias"),
        "use_rslora": adapter.get("use_rslora"),
        "weight_files": [str(item) for item in weight_files],
        "size_bytes": total,
        "size_human": human_bytes(total),
        "trainable_parameters": (report.get("weights") or {}).get("parameter_count"),
        "architecture": (report.get("architecture") or {}).get("model_type"),
        "dtype": sorted((report.get("weights") or {}).get("dtypes", {}).keys()),
        "raw_config": models.read_json(root / "adapter_config.json", {}),
    }


def scan(roots: list[str], max_depth: int = 3) -> list[dict[str, Any]]:
    """Walk the given roots and describe every adapter folder found."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in roots:
        root = Path(raw).expanduser()
        if not root.exists():
            continue
        candidates = [root] if is_adapter(root) else []
        base_depth = len(root.parts)
        for path in root.rglob("adapter_config.json"):
            if len(path.parts) - base_depth > max_depth:
                continue
            candidates.append(path.parent)
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                found.append(describe(candidate))
            except ZxError as exc:
                found.append({"path": str(candidate), "name": candidate.name, "error": exc.to_dict()})
    found.sort(key=lambda item: item.get("name", ""))
    return found


def merge_plan(adapter_path: str | Path, base_model: str | Path | None = None,
               destination: str | Path | None = None) -> dict[str, Any]:
    """Compatibility report for merging an adapter into its base model."""
    from .backends import hf

    info = describe(adapter_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, status: str, message: str, hint: str = "") -> None:
        checks.append({"name": name, "status": status, "message": message, "hint": hint})

    target_base = str(base_model or info["base_model"] or "")
    if not target_base:
        check("base model", "Unsupported",
              "The adapter config does not name a base model.",
              "Pick the base model explicitly before merging.")
    elif not Path(target_base).exists():
        check("base model", "Unsupported",
              f"Base model path does not exist: {target_base}",
              "Point at a local copy of the base model.")
    else:
        base_report = models.inspect(target_base)
        check("base model", "Supported", f"Base model found: {Path(target_base).name}")
        base_params = (base_report.get("weights") or {}).get("parameter_count")
        adapter_params = info.get("trainable_parameters")
        if base_params and adapter_params and adapter_params > base_params:
            check("parameter count", "Experimental",
                  "The adapter holds more parameters than the base model — check that the right pair was selected.")
        else:
            check("parameter count", "Supported",
                  f"Adapter adds {adapter_params or '?'} parameters on top of {base_params or '?'}.")

    available, reason, requires = hf.availability()
    check("merge backend", "Supported" if available else "Unsupported",
          "The PyTorch runtime is available for merging." if available else reason,
          "" if available else f"Missing: {', '.join(requires)}")

    statuses = {entry["status"] for entry in checks}
    status = "Unsupported" if "Unsupported" in statuses else ("Supported with limitations" if "Experimental" in statuses else "Supported")
    return {
        "adapter": info,
        "base_model": target_base or None,
        "destination": str(destination) if destination else None,
        "checks": checks,
        "status": status,
        "backend_available": available,
        "notes": [
            "Merging writes a new model folder; the base model and the adapter are left untouched.",
            "Merged weights can no longer be unmerged unless the adapter is kept.",
        ],
    }
