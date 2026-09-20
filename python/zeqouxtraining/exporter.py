"""Export a trained artefact to a folder the user chooses.

Two modes, and the difference matters:

* **copy** — always possible. The adapter (or full model) files are copied
  together with a generated `README.md` and `export.json`, so the folder is
  self-describing: which base model it belongs to, how it was trained, and what
  the loss was. This is what most users want.
* **merge** — folds the adapter into the base model so the result is a
  standalone model with no PEFT dependency. That needs `torch` and `peft`
  installed, because the weights genuinely have to be combined. When they are
  missing the command says so instead of writing a folder that would not load.

Merging is never faked: if the runtime is absent, the caller is told exactly
which packages to install.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import events

#: Files that make up the trainable artefact, in the order they are copied.
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
#: Everything else worth carrying along, matched by suffix.
COMPANION_SUFFIXES = (
    ".json", ".txt", ".model", ".vocab", ".merges", ".tiktoken",
    ".safetensors", ".bin", ".pt", ".py",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


class ExportError(Exception):
    def __init__(self, message: str, *, hint: str = "", code: str = "export_error") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - a missing or broken file is not fatal here
        return None


def _dir_size(target: Path) -> int:
    total = 0
    for path in target.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def runtime() -> dict[str, Any]:
    """What merging would need, and whether it is present."""
    missing = [name for name in ("torch", "peft", "transformers") if not _available(name)]
    return {
        "available": not missing,
        "missing": missing,
        "hint": (
            ""
            if not missing
            else "Install the ML runtime in Settings → Environment, then merge again."
        ),
    }


def describe(source_dir: str | Path) -> dict[str, Any]:
    """Inspect an artefact folder and report which export modes apply."""
    source = Path(str(source_dir)).expanduser()
    if not source.exists():
        raise ExportError(
            f"'{source}' does not exist.",
            hint="Pick a trained run folder from the Models or Training screen.",
            code="not_found",
        )
    if not source.is_dir():
        raise ExportError(
            f"'{source}' is a file, not a folder.",
            hint="Exporting takes the whole run folder.",
            code="not_a_directory",
        )

    files = sorted(path.name for path in source.iterdir() if path.is_file())
    weights = [name for name in files if name.endswith(WEIGHT_SUFFIXES)]
    has_weights = bool(weights)

    adapter_config = _read_json(source / "adapter_config.json")
    is_adapter = adapter_config is not None
    base_model = None
    if adapter_config:
        base_model = adapter_config.get("base_model_name_or_path")
    elif (model_config := _read_json(source / "config.json")):
        base_model = model_config.get("_name_or_path") or None

    checkpoints = sorted(
        path.name for path in source.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    )

    copy_ready = has_weights
    merge_ready = is_adapter and has_weights
    merge_runtime = runtime()
    blockers: list[dict[str, str]] = []
    if not has_weights:
        blockers.append({
            "code": "no_weights",
            "message": "This folder holds no weight files.",
            "hint": "Run or resume training until at least one checkpoint is written.",
        })
    elif merge_ready and not merge_runtime["available"]:
        blockers.append({
            "code": "missing_dependency",
            "message": f"Merging needs: {', '.join(merge_runtime['missing'])}.",
            "hint": merge_runtime["hint"],
        })

    return {
        "path": str(source),
        "name": source.name,
        "files": files,
        "weight_files": weights,
        "is_adapter": is_adapter,
        "adapter_config": adapter_config,
        "base_model": base_model,
        "checkpoints": checkpoints,
        "size_bytes": _dir_size(source),
        "modes": {
            "copy": {"ready": copy_ready},
            "merge": {
                "ready": merge_ready and merge_runtime["available"],
                "applicable": merge_ready,
                "missing": merge_runtime["missing"],
            },
        },
        "blockers": blockers,
    }


def _readme(info: dict[str, Any], metadata: dict[str, Any], mode: str) -> str:
    """A human-readable card that travels with the exported files."""
    base = metadata.get("baseModel") or info.get("base_model") or "unknown"
    lines = [
        f"# {metadata.get('name') or info['name']}",
        "",
        f"Exported from ZeqouXTraining on {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        "| | |",
        "| --- | --- |",
        f"| Mode | `{mode}` |",
        f"| Base model | `{base}` |",
        f"| Method | `{metadata.get('method') or 'unknown'}` |",
    ]
    if metadata.get("datasetName"):
        dataset = metadata["datasetName"]
        if metadata.get("datasetRecords"):
            dataset += f" ({metadata['datasetRecords']} records)"
        lines.append(f"| Dataset | `{dataset}` |")
    if metadata.get("finalLoss") is not None:
        lines.append(f"| Final loss | {metadata['finalLoss']} |")
    if metadata.get("steps"):
        lines.append(f"| Steps | {metadata['steps']} |")
    if info.get("base_model") and info.get("adapter_config"):
        config = info["adapter_config"]
        lines.append(
            f"| LoRA | r={config.get('r')} · alpha={config.get('lora_alpha')} · "
            f"dropout={config.get('lora_dropout')} |"
        )
    lines += ["", "## Files", ""]
    for name in info["files"]:
        lines.append(f"- `{name}`")
    lines.append("")
    if mode == "merge":
        # json.dumps quotes the path safely, which matters on Windows.
        where = json.dumps(str(metadata.get("outputDir") or "."))
        lines += [
            "## Usage",
            "",
            "The adapter was merged into the base model, so this folder is a",
            "standalone model:",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f"model = AutoModelForCausalLM.from_pretrained({where})",
            f"tokenizer = AutoTokenizer.from_pretrained({where})",
            "```",
        ]
    else:
        lines += [
            "## Usage",
            "",
            "This is a PEFT adapter. Load it on top of its base model:",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "from peft import PeftModel",
            "",
            f"base = AutoModelForCausalLM.from_pretrained({json.dumps(str(base))})",
            "model = PeftModel.from_pretrained(base, '.')",
            "tokenizer = AutoTokenizer.from_pretrained('.')",
            "```",
            "",
            "The base model is not included — only the trained adapter is.",
        ]
    lines += [
        "",
        "## Provenance",
        "",
        "`export.json` in this folder holds the full record: the run id, the",
        "configuration that produced these weights, and the training metrics.",
        "",
    ]
    return "\n".join(lines)


def export(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    merge: bool = False,
    base_model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy — and optionally merge — a trained artefact into `output_dir`."""
    info = describe(source_dir)
    source = Path(info["path"])
    target = Path(str(output_dir)).expanduser()
    meta = dict(metadata or {})
    meta.setdefault("outputDir", str(target))

    # Decide everything that can fail *before* touching the filesystem, so a
    # refusal never leaves an empty folder behind.
    if not info["weight_files"]:
        blocker = info["blockers"][0]
        raise ExportError(blocker["message"], hint=blocker["hint"], code=blocker["code"])

    if merge:
        if not info["modes"]["merge"]["applicable"]:
            raise ExportError(
                "This artefact has no adapter to merge.",
                hint="Merging applies to LoRA/QLoRA adapters. Copy the folder as-is instead.",
                code="not_an_adapter",
            )
        missing = info["modes"]["merge"]["missing"]
        if missing:
            raise ExportError(
                f"Merging needs: {', '.join(missing)}.",
                hint="Install the ML runtime in Settings → Environment, then merge again.",
                code="missing_dependency",
            )
        mode = "merge"
    else:
        mode = "copy"

    if target.exists() and any(target.iterdir()):
        raise ExportError(
            f"'{target}' is not empty.",
            hint="Choose an empty folder, or make a new one — nothing is written over.",
            code="not_empty",
        )
    target.mkdir(parents=True, exist_ok=True)

    started = time.time()
    events.stage("export", f"Exporting to {target}")

    if mode == "merge":
        events.log("Loading the base model and folding in the adapter…")
        _merge_into(source, target, base_model or info.get("base_model"), meta)
    else:
        copied = _copy_artefact(source, target)
        events.log(f"Copied {len(copied)} files.")

    # Provenance and the human-readable card, written last so their presence
    # means the export completed.
    payload = {
        "exported_at": time.time(),
        "exported_by": "ZeqouXTraining",
        "mode": mode,
        "source": str(source),
        "base_model": base_model or info.get("base_model"),
        "adapter": info["adapter_config"],
        "run": meta,
    }
    (target / "export.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (target / "README.md").write_text(_readme(info, meta, mode), encoding="utf-8")

    return {
        "mode": mode,
        "output_dir": str(target),
        "source_dir": str(source),
        "files": sorted(path.name for path in target.iterdir()),
        "size_bytes": _dir_size(target),
        "seconds": round(time.time() - started, 2),
        "base_model": base_model or info.get("base_model"),
        "merged": mode == "merge",
    }


def _copy_artefact(source: Path, target: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        # Everything except an old export manifest the folder might already hold.
        if path.name in ("export.json", "README.md"):
            continue
        if path.name in ADAPTER_FILES or path.name.endswith(COMPANION_SUFFIXES):
            shutil.copy2(path, target / path.name)
            copied.append(path.name)
    return copied


def _merge_into(source: Path, target: Path, base_model: str | None, meta: dict[str, Any]) -> None:
    """Merge the adapter into its base model and save a standalone model."""
    import torch  # noqa: PLC0415 - only reachable once the runtime is installed
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if not base_model:
        raise ExportError(
            "The base model is unknown, so the adapter cannot be merged.",
            hint="Re-open the run in Training so its base model is recorded, or export a copy instead.",
            code="no_base_model",
        )

    events.log(f"Loading base model {base_model}…")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=bool(meta.get("trustRemoteCode")),
    )
    model = PeftModel.from_pretrained(base, str(source))
    events.log("Merging adapter weights…")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(target), safe_serialization=True)
    del base, model, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # A merged model still needs its tokenizer to be usable.
    tokenizer_source = source if (source / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    tokenizer.save_pretrained(str(target))
