"""Model cards written from real run metadata — no invented numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import inspect as inspect_model
from .util import human_bytes, now_iso, read_json


def model_card(model_path: str | Path, lineage: dict[str, Any] | None = None,
               evaluation: dict[str, Any] | None = None, notes: str = "") -> str:
    path = Path(model_path)
    report = inspect_model(path)
    architecture = report.get("architecture") or {}
    weights = report.get("weights") or {}
    run = read_json(path / "zxtrain-run.json", {}) or {}
    spec = (run.get("spec") or {})
    adapter = report.get("adapter_details") or {}
    lines: list[str] = [
        f"# Model card — {report.get('name')}",
        "",
        f"- Path: `{report.get('path')}`",
        f"- Format: `{(report.get('format') or {}).get('kind')}`",
        f"- Size on disk: {human_bytes(report.get('size_bytes'))}",
        f"- Parameters: {weights.get('parameter_count'):,}" if weights.get("parameter_count") else
        "- Parameters: unknown (no readable weight headers)",
        f"- Parameter source: {weights.get('source')}" if weights.get("source") else "",
        f"- Architecture: {architecture.get('model_type')} ({', '.join(architecture.get('architectures') or []) or 'unknown'})",
        f"- Layers: {architecture.get('num_layers')}, hidden size: {architecture.get('hidden_size')}, "
        f"heads: {architecture.get('num_attention_heads')} (KV heads: {architecture.get('num_key_value_heads')})",
        f"- Context window: {architecture.get('max_position_embeddings')}",
        f"- Vocabulary: {architecture.get('vocab_size')}",
        f"- Weight dtypes: {', '.join(f'{key}: {value:,}' for key, value in (weights.get('dtypes') or {}).items()) or 'unknown'}",
    ]
    if adapter:
        lines += [
            "",
            "## Adapter (LoRA/PEFT)",
            "",
            f"- Type: {adapter.get('peft_type')}, task: {adapter.get('task_type')}",
            f"- Rank: {adapter.get('r')}, alpha: {adapter.get('lora_alpha')}, dropout: {adapter.get('lora_dropout')}",
            f"- Target modules: {', '.join(adapter.get('target_modules') or [])}",
            f"- Base model: `{adapter.get('base_model')}`",
        ]
    lines += ["", "## Training", ""]
    if run:
        lines += [
            f"- Method: {run.get('method')}",
            f"- Backend: {run.get('backend')}",
            f"- Base model: `{run.get('base_model')}`",
            f"- Fine-tuning note: {run.get('adapter_note') or 'n/a'}",
            f"- Learning rate: {spec.get('learning_rate')}, epochs: {spec.get('epochs')}, "
            f"batch: {spec.get('batch_size')} × {spec.get('gradient_accumulation')} accumulation",
            f"- Precision: {spec.get('precision')}, quantisation: {spec.get('quantization')}",
            f"- Sequence length: {spec.get('sequence_length')}",
            f"- Seed: {spec.get('seed')}",
            f"- Datasets: {', '.join('`' + str(item) + '`' for item in spec.get('dataset_paths') or [])}",
        ]
        metrics = run.get("metrics") or {}
        if metrics:
            lines.append(
                "- Measured metrics: " + ", ".join(f"{key} = {value:.4f}" for key, value in metrics.items())
            )
    else:
        lines.append("- No training run metadata is stored next to this model "
                     "(it is an imported or base model).")
    if lineage:
        parent = lineage.get("parent") or {}
        lines += [
            "",
            "## Lineage",
            "",
            f"- Parent ({parent.get('kind') or 'unknown'}): `{parent.get('path') or 'none'}`",
            f"- Parent job: {parent.get('job_id') or 'n/a'}",
            f"- Resumed from a checkpoint: {'yes' if lineage.get('resumed') else 'no'}",
            f"- Restored: {', '.join(key for key, value in (lineage.get('restored') or {}).items() if value) or 'weights only'}",
        ]
    if evaluation:
        lines += ["", "## Evaluation", ""]
        metrics = evaluation.get("metrics") or {}
        if metrics:
            lines += [f"- {key}: {value}" for key, value in metrics.items()]
        lines.append(f"- Evaluated on: {', '.join(evaluation.get('datasets') or []) or 'n/a'}")
        lines.append(f"- Evaluated tokens: {evaluation.get('evaluated_tokens')}")
    lines += [
        "",
        "## Intended use and limitations",
        "",
        notes.strip() or "Not documented yet — add notes before sharing this model.",
        "",
        "## Reproducibility",
        "",
        "The full configuration, seed and dataset paths for this model are stored in "
        "`zxtrain-run.json` next to the weights. Export a reproducibility bundle from the app to "
        "capture package versions and hardware as well.",
        "",
        f"Generated at {now_iso()} by ZeqouXTraining.",
    ]
    return "\n".join(line for line in lines if line is not None)


def dataset_card_markdown(report: dict[str, Any], notes: str = "") -> str:
    lines = [
        f"# Dataset card — {report.get('name')}",
        "",
        f"- Path: `{report.get('path')}`",
        f"- Records: {report.get('record_count'):,}",
        f"- Size: {report.get('size_human')}",
        f"- Fields: {', '.join(report.get('field_names') or [])}",
        f"- Duplicate records: {report.get('duplicates')}",
        f"- Empty records: {report.get('empty_records')}",
        f"- Average length: {report.get('length', {}).get('average')} characters",
        f"- Estimated tokens: {(report.get('token_estimate') or {}).get('total')} "
        f"({(report.get('token_estimate') or {}).get('basis', 'heuristic')})",
        "",
        "## Mapping",
        "",
        f"- Detected: {report.get('detected_mapping')}",
        "",
        "## Notes and known limitations",
        "",
        notes.strip() or "Not documented yet.",
        "",
        f"Generated at {now_iso()} by ZeqouXTraining.",
    ]
    return "\n".join(lines)
