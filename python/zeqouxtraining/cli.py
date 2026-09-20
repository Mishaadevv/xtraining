"""Command-line entry points.

The desktop app drives the backend through this module:

    python -m zeqouxtraining.cli env-check
    python -m zeqouxtraining.cli hardware
    python -m zeqouxtraining.cli validate-dataset --path ... --context-length 512
    python -m zeqouxtraining.cli inspect-model --source ...
    python -m zeqouxtraining.cli auto-config --base-model ... --dataset-path ...
    python -m zeqouxtraining.cli train --job <job.json>
    python -m zeqouxtraining.cli checkpoints --run-dir ...
    python -m zeqouxtraining.cli infer
    python -m zeqouxtraining.cli backends
    python -m zeqouxtraining.cli install-plan --cuda-tag cu124

Only ``train`` and ``infer`` are long-lived; the rest answer once with a
``result`` event.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import events


def _json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--json argument is not valid JSON: {exc}") from exc


def _guard(fn, stage: str) -> int:
    """Run a one-shot command, reporting failures as protocol events."""
    from .errors import humanize

    try:
        payload = fn()
    except SystemExit:
        raise
    except Exception as exc:
        info = humanize(exc, stage=stage)
        events.fail(info["message"], hint=info["hint"], code=info["code"],
                    traceback_text=info["traceback"])
        return 1
    events.result(payload)
    return 0


# --------------------------------------------------------------------------- #
# One-shot commands
# --------------------------------------------------------------------------- #

def cmd_env_check(_args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from . import deps, hardware
        from .backends.registry import backend_capabilities

        return {
            "dependencies": deps.inspect(),
            "hardware": hardware.detect(),
            "backends": backend_capabilities(),
            "install_plan": deps.install_plan(),
        }

    return _guard(run, "env-check")


def cmd_hardware(_args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from . import hardware

        return hardware.detect()

    return _guard(run, "hardware")


def cmd_backends(_args: argparse.Namespace) -> int:
    from .backends.registry import backend_capabilities

    events.result({"backends": backend_capabilities()})
    return 0


def cmd_install_plan(args: argparse.Namespace) -> int:
    from . import deps

    events.result(deps.install_plan(args.cuda_tag))
    return 0


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from .datasets import validate

        def progress(message: str, index: int, total: int) -> None:
            events.emit("dataset-progress", {"message": message, "index": index, "total": total})

        return validate(
            args.path,
            hf_id=args.hf_id,
            split=args.split,
            mapping=_json_arg(args.mapping),
            context_length=args.context_length,
            max_records=args.max_records,
            preview_limit=args.preview,
            progress=progress,
        )

    return _guard(run, "validate-dataset")


def cmd_preview_dataset(args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from .datasets import detect_mapping, normalize, resolve_source, sample_text

        records, meta = resolve_source(
            path=args.path,
            hf_id=args.hf_id,
            split=args.split,
            fmt=args.format,
            max_records=args.max_records,
        )
        mapping = _json_arg(args.mapping) or detect_mapping(records)
        samples = normalize(records, mapping)
        return {
            "source": meta,
            "mapping": mapping,
            "count": len(samples),
            "samples": [sample_text(sample)[:2000] for sample in samples[: args.limit]],
        }

    return _guard(run, "preview-dataset")


def cmd_inspect_model(args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from .models import inspect

        return inspect(args.source, hf_cache_dir=args.hf_cache)

    return _guard(run, "inspect-model")


def cmd_auto_config(args: argparse.Namespace) -> int:
    """Hardware + model + dataset -> a safe configuration, with reasons."""
    def run() -> dict[str, Any]:
        from . import hardware
        from .config import auto_configure, baseline, validate
        from .datasets import validate as validate_dataset
        from .estimator import estimate
        from .models import inspect

        detected = hardware.detect()

        model_info: dict[str, Any] = {}
        model_error = None
        if args.base_model:
            try:
                model_info = inspect(args.base_model, hf_cache_dir=args.hf_cache)
            except Exception as exc:  # surfaced, not fatal: the user can still see hardware info
                model_error = str(exc)

        dataset_report = None
        if args.dataset_path:
            dataset_report = validate_dataset(
                args.dataset_path,
                context_length=args.context_length,
                max_records=2000,
                preview_limit=0,
            )

        # With automatic configuration off the user still needs a valid starting
        # point, but it must not be presented as hardware-derived.
        suggested = baseline() if args.baseline else auto_configure(detected, model_info, dataset_report)
        config = suggested["config"]
        # auto_configure only patches tuning values; carry over the actual
        # selection so validation reports on this run, not on an empty config.
        config["base_model"] = args.base_model or ""
        config["dataset"] = {
            **(config.get("dataset") or {}),
            "path": args.dataset_path or "",
        }
        if args.method:
            config["method"] = args.method
            if config["method"] == "qlora":
                config["quantization"] = "4bit"

        devices = (detected.get("cuda", {}).get("devices") or [])
        available_vram = float(devices[0].get("total_memory_mb") or 0) if devices else None
        if not available_vram:
            gpus = detected.get("gpu", {}).get("gpus") or []
            available_vram = float(gpus[0].get("memory_total_mb") or 0) if gpus else None
        memory = detected.get("memory") or {}
        available_ram = float(memory.get("total_mb") or 0) or None

        return {
            "hardware": detected,
            "model": model_info,
            "model_error": model_error,
            "dataset_report": dataset_report,
            "config": config,
            "reasons": suggested["reasons"],
            "estimate": estimate(config, model_info, available_vram, available_ram),
            "issues": validate(config, detected),
        }

    return _guard(run, "auto-config")


def cmd_checkpoints(args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from .checkpoints import list_checkpoints, size_of

        return {
            "run_dir": args.run_dir,
            "checkpoints": list_checkpoints(args.run_dir),
            "size_bytes": size_of(args.run_dir),
        }

    return _guard(run, "checkpoints")


def cmd_export(args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from .exporter import describe, export

        if args.describe:
            return {"info": describe(args.source)}
        if not args.output:
            raise ValueError("--output is required unless --describe is used.")
        return export(
            args.source,
            args.output,
            merge=args.merge,
            base_model=args.base_model,
            metadata=_json_arg(args.metadata),
        )

    return _guard(run, "export")


def cmd_estimate(args: argparse.Namespace) -> int:
    def run() -> dict[str, Any]:
        from .config import normalize
        from .estimator import estimate
        from .models import inspect

        config = normalize(_json_arg(args.config) or {})
        model_info = _json_arg(args.model_info)
        if model_info is None and config.get("base_model"):
            model_info = inspect(str(config["base_model"]))
        ram = args.available_ram
        if ram is None:
            try:
                from .hardware import detect
                ram = float((detect().get("memory") or {}).get("total_mb") or 0) or None
            except Exception:
                ram = None
        return estimate(config, model_info or {}, args.available_vram, ram)

    return _guard(run, "estimate")


def cmd_train(args: argparse.Namespace) -> int:
    from .trainer import EXIT_ERROR, run_job

    try:
        return run_job(args.job)
    except Exception as exc:
        from .errors import humanize

        info = humanize(exc, stage="job")
        events.fail(info["message"], hint=info["hint"], code=info["code"],
                    traceback_text=info["traceback"])
        return EXIT_ERROR


def cmd_infer(_args: argparse.Namespace) -> int:
    from .inference import serve

    return serve()


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zeqouxtraining",
        description="ZeqouXTraining backend — hardware, datasets, training and inference.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env-check", help="Dependencies, hardware and backend availability")
    sub.add_parser("hardware", help="Hardware and CUDA snapshot")
    sub.add_parser("backends", help="Training backend capabilities")

    install = sub.add_parser("install-plan", help="pip command that installs the ML runtime")
    install.add_argument("--cuda-tag", default=None,
                         help="CUDA wheel tag, e.g. cu124 (omit for a CPU build)")

    validate = sub.add_parser("validate-dataset", help="Validate a dataset and report issues")
    validate.add_argument("--path", default=None, help="Local file or folder of shards")
    validate.add_argument("--hf-id", default=None, help="Hugging Face dataset id, e.g. tatsu-lab/alpaca")
    validate.add_argument("--split", default="train", help="Split used when --hf-id is given")
    validate.add_argument("--format", default="auto")
    validate.add_argument("--mapping", default=None, help="JSON field mapping")
    validate.add_argument("--context-length", type=int, default=512)
    validate.add_argument("--max-records", type=int, default=20000)
    validate.add_argument("--preview", type=int, default=8)

    preview = sub.add_parser("preview-dataset", help="Show the first normalised samples")
    preview.add_argument("--path", default=None, help="Local file or folder of shards")
    preview.add_argument("--hf-id", default=None, help="Hugging Face dataset id, e.g. tatsu-lab/alpaca")
    preview.add_argument("--split", default="train", help="Split used when --hf-id is given")
    preview.add_argument("--format", default="auto")
    preview.add_argument("--mapping", default=None)
    preview.add_argument("--limit", type=int, default=5)
    preview.add_argument("--max-records", type=int, default=500)

    inspect = sub.add_parser("inspect-model", help="Inspect a local or Hugging Face model")
    inspect.add_argument("--source", required=True)
    inspect.add_argument("--hf-cache", default=None)

    auto = sub.add_parser("auto-config", help="Pick safe parameters from the real hardware")
    auto.add_argument("--base-model", default=None)
    auto.add_argument("--dataset-path", default=None)
    auto.add_argument("--context-length", type=int, default=512)
    auto.add_argument("--method", default=None)
    auto.add_argument("--hf-cache", default=None)
    auto.add_argument("--baseline", action="store_true",
                      help="Return the documented defaults instead of hardware-derived values")

    checkpoints = sub.add_parser("checkpoints", help="List checkpoints for a run directory")
    checkpoints.add_argument("--run-dir", required=True)

    export = sub.add_parser("export", help="Copy, or merge, a trained artefact")
    export.add_argument("--source", required=True, help="Trained run or checkpoint folder")
    export.add_argument("--output", default=None, help="Destination folder (required unless --describe)")
    export.add_argument("--merge", action="store_true",
                        help="Fold the adapter into the base model (needs torch + peft)")
    export.add_argument("--base-model", default=None, help="Override the base model id")
    export.add_argument("--metadata", default=None, help="JSON run metadata for provenance")
    export.add_argument("--describe", action="store_true",
                        help="Report what this folder holds instead of exporting")

    estimate = sub.add_parser("estimate", help="Estimate VRAM for a configuration")
    estimate.add_argument("--config", default=None, help="JSON training config")
    estimate.add_argument("--model-info", default=None, help="JSON model info")
    estimate.add_argument("--available-vram", type=float, default=None, help="VRAM in MB")
    estimate.add_argument("--available-ram", type=float, default=None, help="System RAM in MB")

    train = sub.add_parser("train", help="Run a training job")
    train.add_argument("--job", required=True)

    sub.add_parser("infer", help="Serve inference requests over stdin/stdout")

    return parser


HANDLERS = {
    "env-check": cmd_env_check,
    "hardware": cmd_hardware,
    "backends": cmd_backends,
    "install-plan": cmd_install_plan,
    "validate-dataset": cmd_validate_dataset,
    "preview-dataset": cmd_preview_dataset,
    "export": cmd_export,
    "inspect-model": cmd_inspect_model,
    "auto-config": cmd_auto_config,
    "checkpoints": cmd_checkpoints,
    "estimate": cmd_estimate,
    "train": cmd_train,
    "infer": cmd_infer,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Training and inference stream for a long time; keep library output away
    # from the protocol before any library is imported.
    if args.command in ("train", "infer"):
        events.install_diagnostics_redirect()

    handler = HANDLERS.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
