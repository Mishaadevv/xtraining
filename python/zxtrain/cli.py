"""Engine entry point.

Usage (all payloads are JSON on stdin, one JSON object on stdout):

    python -m zxtrain.cli <method>
    python -m zxtrain.cli run <spec.json>     # long running job, streams @@event lines
    python -m zxtrain.cli serve               # inference sidecar, JSON line protocol

The Electron main process uses `call`, `run` and `serve`. Both the GUI and any
CLI/automation therefore share exactly the same engine.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):  # allow `python zxtrain/cli.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zxtrain import ENGINE_VERSION, PROTOCOL_VERSION  # noqa: E402
from zxtrain import datasets as ds  # noqa: E402
from zxtrain import estimator, environment, hardware, storage  # noqa: E402
from zxtrain import models as model_tools  # noqa: E402
from zxtrain import precision as precision_tools  # noqa: E402
from zxtrain import trainer  # noqa: E402
from zxtrain.backends import registry  # noqa: E402
from zxtrain.errors import ZxError, diagnose, from_exception  # noqa: E402
from zxtrain.util import json_safe, read_json, write_json_atomic  # noqa: E402


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ZxError(
            code="bad_payload",
            message=f"Request payload is not valid JSON: {exc.msg}",
            hint="The UI sends JSON on stdin; check the request body.",
        ) from exc
    if not isinstance(payload, dict):
        raise ZxError(code="bad_payload", message="Request payload must be a JSON object.")
    return payload


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #

def m_health(_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": "zxtrain",
        "version": ENGINE_VERSION,
        "protocol": PROTOCOL_VERSION,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pid": os.getpid(),
        "platform": sys.platform,
    }


def m_engine_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = payload.get("workspace") or str(Path.home())
    return {
        "hardware": hardware.detect([workspace]),
        "hardware_live": hardware.read_live_resources(),
        "backends": registry.describe(),
        "methods": registry.supported_methods(),
        "environment": environment.report(workspace),
        "storage": storage.report(workspace),
    }


def m_hardware_detect(payload: dict[str, Any]) -> dict[str, Any]:
    return hardware.detect(payload.get("storage_paths") or None)


def m_hardware_live(_payload: dict[str, Any]) -> dict[str, Any]:
    return hardware.read_live_resources()


def m_env_report(payload: dict[str, Any]) -> dict[str, Any]:
    return environment.report(payload.get("workspace") or str(Path.home()))


def m_env_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return environment.install_plan(
        payload.get("workspace") or str(Path.home()),
        cuda=bool(payload.get("cuda")),
        extra=payload.get("extra"),
    )


def m_backends_list(_payload: dict[str, Any]) -> dict[str, Any]:
    return {"backends": registry.describe(), "methods": registry.supported_methods()}


def m_models_scan(payload: dict[str, Any]) -> dict[str, Any]:
    roots = payload.get("roots") or []
    if not roots:
        raise ZxError(code="no_roots", message="No folders were given to scan.",
                      hint="Add a models folder in Settings, or import a model directly.")
    return {"models": model_tools.scan(roots, int(payload.get("max_depth") or 3))}


def m_models_inspect(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="models.inspect needs a path.")
    return model_tools.inspect(path, with_digest=bool(payload.get("digest")))


def m_models_import(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    destination = payload.get("destination")
    if not source or not destination:
        raise ZxError(code="bad_payload", message="models.import needs source and destination.")
    target = model_tools.import_into(source, Path(destination), copy=bool(payload.get("copy", True)),
                                     name=payload.get("name"))
    return {"imported": str(target), "inspection": model_tools.inspect(target)}


def m_models_card(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain import cards

    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="models.card needs a path.")
    markdown = cards.model_card(path, payload.get("lineage"), payload.get("evaluation"),
                               payload.get("notes") or "")
    if payload.get("write"):
        write_json_atomic(Path(path) / "model-card.json", {"markdown": markdown})
        (Path(path) / "README.md").write_text(markdown, encoding="utf-8")
    return {"markdown": markdown, "written": bool(payload.get("write"))}


def m_datasets_inspect(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="datasets.inspect needs a path.")
    return ds.inspect_dataset(path, payload.get("mapping") or {},
                              int(payload.get("sample_size") or 4000),
                              float(payload.get("near_duplicate_threshold", 0.9)))


def m_datasets_preview(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="datasets.preview needs a path.")
    mapping = payload.get("mapping") or {}
    limit = int(payload.get("limit") or 50)
    offset = int(payload.get("offset") or 0)
    template = payload.get("template") or "chatml"
    rows = []
    for index, record in enumerate(ds.iter_records(Path(path), limit=limit, offset=offset)):
        rows.append({
            "index": offset + index,
            "record": json_safe(record),
            "normalised": json_safe(ds.normalise_record(record, mapping)),
            "text": ds.record_to_text(record, mapping, template),
        })
    return {"rows": rows, "offset": offset, "limit": limit, "mapping": mapping}


def m_datasets_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="datasets.mapping needs a path.")
    fields: list[str] = []
    sample: dict[str, Any] | None = None
    for record in ds.iter_records(Path(path), limit=5):
        if sample is None:
            sample = record
        for key in record:
            if key not in fields:
                fields.append(key)
    return ds.detect_mapping(fields, sample)


def m_datasets_plan_clean(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="datasets.plan_clean needs a path.")
    return ds.plan_cleaning(Path(path), payload.get("mapping") or {},
                            payload.get("operations") or [],
                            int(payload.get("preview_limit") or 8))


def m_datasets_clean(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    destination = payload.get("destination")
    if not path or not destination:
        raise ZxError(code="bad_payload", message="datasets.clean needs path and destination.")
    target = Path(str(destination))
    if target.exists() and not payload.get("overwrite"):
        stamp = f"{target.stem}-{os.getpid()}{target.suffix}"
        target = target.with_name(stamp)
    return ds.clean_dataset(Path(path), payload.get("mapping") or {},
                            payload.get("operations") or [], target,
                            payload.get("format") or "jsonl")


def m_datasets_split(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    output_dir = payload.get("output_dir")
    if not path or not output_dir:
        raise ZxError(code="bad_payload", message="datasets.split needs path and output_dir.")
    return ds.split_dataset(Path(path), payload.get("mapping") or {}, Path(str(output_dir)),
                            payload.get("ratios"), payload.get("counts"),
                            int(payload.get("seed") or 42), payload.get("format") or "jsonl",
                            payload.get("grouped_by"))


def m_datasets_export(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    destination = payload.get("destination")
    if not path or not destination:
        raise ZxError(code="bad_payload", message="datasets.export needs path and destination.")
    return ds.export_dataset(Path(path), payload.get("mapping") or {}, Path(str(destination)),
                             payload.get("format") or "jsonl")


def m_datasets_from_conversations(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn playground conversations or feedback records into a real dataset file."""
    destination = payload.get("destination")
    if not destination:
        raise ZxError(code="bad_payload", message="datasets.from_conversations needs a destination.")
    conversations = payload.get("conversations") or []
    records: list[dict[str, Any]] = []
    for conversation in conversations:
        if not isinstance(conversation, dict):
            continue
        messages = conversation.get("messages")
        if isinstance(messages, list) and messages:
            cleaned = [
                {"role": str(message.get("role")), "content": str(message.get("content", ""))}
                for message in messages
                if isinstance(message, dict) and str(message.get("content", "")).strip()
            ]
            if len(cleaned) >= 2:
                records.append({"messages": cleaned})
        elif conversation.get("text"):
            records.append({"text": str(conversation["text"])})
    if not records:
        raise ZxError(
            code="dataset_empty",
            message="No usable conversations were supplied.",
            hint="Keep at least one user message and one assistant reply before exporting.",
        )
    target = Path(str(destination))
    ds.write_records(records, target, payload.get("format") or "jsonl")
    return {"output": str(target), "records": len(records), "format": payload.get("format") or "jsonl"}


def m_datasets_card(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain import cards

    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="datasets.card needs a path.")
    report = payload.get("report") or ds.inspect_dataset(path, payload.get("mapping") or {}, 1000)
    markdown = cards.dataset_card_markdown(report, payload.get("notes") or "")
    if payload.get("write"):
        (Path(str(path)).parent / f"{Path(str(path)).name}.card.md").write_text(markdown, encoding="utf-8")
    return {"markdown": markdown, "written": bool(payload.get("write"))}


def m_tokenizer_encode(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain.bpe import BPETokenizer

    path = payload.get("tokenizer_path")
    if not path or not Path(str(path)).exists():
        raise ZxError(
            code="tokenizer_missing",
            message="No tokenizer file available for this comparison.",
            hint="Train a tokenizer on a dataset, or pick a model folder that ships tokenizer.json.",
        )
    tokenizer = BPETokenizer.load(Path(str(path)))
    text = str(payload.get("text") or "")
    ids = tokenizer.encode(text)
    return {
        "tokens": ids,
        "pieces": [tokenizer.decode([token]) for token in ids],
        "count": len(ids),
        "vocab_size": tokenizer.vocab_size,
        "decoded": tokenizer.decode(ids),
    }


def m_tokenizer_decode(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(Path(str(payload.get("tokenizer_path"))))
    return {"text": tokenizer.decode(payload.get("tokens") or [])}


def m_tokenizer_stats(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain.bpe import BPETokenizer, stats_for_texts

    path = payload.get("path")
    if not path:
        raise ZxError(code="bad_payload", message="tokenizer.stats needs a dataset path.")
    mapping = payload.get("mapping") or {}
    texts = ds.extract_texts(Path(path), mapping, limit=int(payload.get("limit") or 2000),
                             template=payload.get("template") or "chatml")
    tokenizer_path = payload.get("tokenizer_path")
    if tokenizer_path and Path(str(tokenizer_path)).exists():
        tokenizer = BPETokenizer.load(Path(str(tokenizer_path)))
        stats = stats_for_texts(tokenizer, texts)
        return {"stats": stats, "tokenizer": str(tokenizer_path), "exact": True}
    characters = [len(text) for text in texts]
    return {
        "stats": {
            "count": len(texts),
            "total_tokens": sum(characters) // 4,
            "average": round(sum(characters) / max(1, len(characters)) / 4, 1),
            "max": max(characters, default=0) // 4,
            "chars_per_token": 4,
        },
        "tokenizer": None,
        "exact": False,
        "note": "Estimates use characters / 4. Train a tokenizer for exact counts.",
    }


def m_training_prepare(payload: dict[str, Any]) -> dict[str, Any]:
    return trainer.prepare(payload)


def m_training_plan(payload: dict[str, Any]) -> dict[str, Any]:
    plan = estimator.plan_run(payload, hardware.detect([payload.get("workspace") or str(Path.home())]))
    return {"plan": plan, "summary": estimator.to_summary(plan)}


def m_auto_configure(payload: dict[str, Any]) -> dict[str, Any]:
    return plan_auto_configuration(payload)


def m_jobs_list(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(payload.get("workspace") or Path.home())
    jobs_root = workspace / "jobs"
    jobs: list[dict[str, Any]] = []
    if jobs_root.exists():
        for job_dir in sorted(jobs_root.iterdir(), reverse=True):
            if job_dir.is_dir():
                summary = trainer.job_summary(job_dir)
                summary["metrics"] = (summary.get("metrics") or [])[-120:]
                jobs.append(summary)
    return {"jobs": jobs}


def m_jobs_summary(payload: dict[str, Any]) -> dict[str, Any]:
    job_dir = payload.get("job_dir")
    if not job_dir:
        raise ZxError(code="bad_payload", message="jobs.summary needs a job_dir.")
    return trainer.job_summary(Path(str(job_dir)))


def m_jobs_reconcile(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(payload.get("workspace") or Path.home())
    return {"findings": trainer.reconcile(workspace / "jobs")}


def m_jobs_logs(payload: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(str(payload.get("job_dir")))
    lines = int(payload.get("lines") or 200)
    event_path = job_dir / "events.jsonl"
    events: list[dict[str, Any]] = []
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "raw", "message": line})
    return {"events": events, "job_dir": str(job_dir)}


def m_lineage_list(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(payload.get("workspace") or Path.home())
    entries = []
    lineage_path = workspace / "jobs" / "lineage.jsonl"
    if lineage_path.exists():
        for line in lineage_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"entries": entries, "roots": _lineage_tree(entries)}


def _lineage_tree(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for entry in entries:
        node_id = entry.get("job_id")
        if not node_id:
            continue
        nodes[node_id] = {
            "id": node_id,
            "method": (entry.get("child") or {}).get("method"),
            "backend": (entry.get("child") or {}).get("backend"),
            "steps": (entry.get("child") or {}).get("steps"),
            "model_dir": (entry.get("child") or {}).get("model_dir"),
            "created_at": entry.get("created_at"),
            "resumed": entry.get("resumed"),
            "parent_path": (entry.get("parent") or {}).get("path"),
            "children": [],
        }
    by_path = {node["model_dir"]: node_id for node_id, node in nodes.items() if node.get("model_dir")}
    roots: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        parent_path = node.get("parent_path")
        parent_id = by_path.get(parent_path)
        if parent_id and parent_id != node_id:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)
    return roots


def m_storage_report(payload: dict[str, Any]) -> dict[str, Any]:
    return storage.report(payload.get("workspace") or str(Path.home()), payload.get("extra_paths"))


def m_storage_tree(payload: dict[str, Any]) -> dict[str, Any]:
    return storage.tree(payload.get("path") or str(Path.home()),
                        int(payload.get("depth") or 2), int(payload.get("max_entries") or 500))


def m_storage_orphans(payload: dict[str, Any]) -> dict[str, Any]:
    return storage.orphans(payload.get("workspace") or str(Path.home()))


def m_storage_clean(payload: dict[str, Any]) -> dict[str, Any]:
    return storage.clean(payload.get("paths") or [], bool(payload.get("confirm")))


def m_storage_protect(payload: dict[str, Any]) -> dict[str, Any]:
    return storage.set_protected(payload.get("path"), bool(payload.get("protected")))


def m_errors_diagnose(payload: dict[str, Any]) -> dict[str, Any]:
    code, needle, advice = diagnose(str(payload.get("text") or ""))
    return {"code": code, "matched": needle, "recommendation": advice}


def m_reproducibility_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(payload.get("workspace") or Path.home())
    job_dir = Path(str(payload.get("job_dir"))) if payload.get("job_dir") else None
    bundle = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": {"name": "zxtrain", "version": ENGINE_VERSION},
        "python": sys.version,
        "hardware": hardware.detect([str(workspace)]),
        "environment": environment.report(workspace),
    }
    if job_dir:
        bundle["job"] = {
            "spec": read_json(job_dir / "spec.json", {}),
            "status": read_json(job_dir / "status.json", {}),
            "lineage": read_json(job_dir / "lineage.json", {}),
        }
    if payload.get("write"):
        out = Path(str(payload.get("output") or (job_dir or workspace) / "reproducibility.json"))
        write_json_atomic(out, bundle)
        bundle["written_to"] = str(out)
    return bundle


def plan_auto_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    """Auto Configure: derive a defensible configuration and explain each choice."""
    report = hardware.detect([payload.get("workspace") or str(Path.home())])
    gpus = report["gpus"]
    capabilities = report["capabilities"]
    vram = int(gpus[0]["memory_total"]) if gpus and gpus[0].get("memory_total") else None
    ram = int(report["memory"]["total"] or 0)
    ram_free = int(report["memory"]["available"] or 0)
    model_params = int(payload.get("parameters") or 0)
    method = str(payload.get("method") or ("qlora" if vram and vram < 12 * 1024**3 else "lora"))
    precision = "bf16" if capabilities["precisions"][2]["available"] else (
        "fp16" if capabilities["precisions"][1]["available"] else "fp32")
    quantization = "none"
    reasons: list[str] = []
    if method == "qlora":
        quantization = "int4"
        reasons.append("4-bit base weights keep the frozen model inside the available VRAM.")
    if method in ("full_finetune", "continued_pretraining") and vram and model_params and vram < model_params * 18:
        method = "lora"
        reasons.append(
            "Full fine-tuning needs roughly 18 bytes per parameter for weights, gradients and AdamW "
            "state; this machine cannot hold that, so LoRA was selected instead.")

    sequence_length = int(payload.get("sequence_length") or 512)
    hidden = int(payload.get("hidden_size") or 1024)
    layers = int(payload.get("num_hidden_layers") or 12)
    # Activation budget: aim for ~70% of the device budget including overhead.
    budget = int((vram * 0.7 if vram else (ram_free or ram) * 0.5) or 2 * 1024**3)
    per_sample = max(1, hidden * layers * sequence_length * 2 * 1.6)
    batch_size = max(1, min(32, int(budget / per_sample) or 1))
    if not batch_size:
        batch_size = 1
    reason_batch = (
        f"{batch_size} sequences × {sequence_length} tokens needs about "
        f"{round(per_sample * batch_size / 1024**3, 2)} GiB of activation memory; the budget on this "
        f"device is {round(budget / 1024**3, 2)} GiB."
    )
    accumulation = 1
    target_tokens_per_step = 32768
    while batch_size * accumulation * sequence_length < target_tokens_per_step and accumulation < 64:
        accumulation += 1
    reasons.append(reason_batch)
    reasons.append(
        f"Gradient accumulation {accumulation} brings the effective batch to "
        f"{batch_size * accumulation} sequences ({batch_size * accumulation * sequence_length} tokens per step)."
    )
    steps_per_epoch = int(payload.get("steps_per_epoch") or 0)
    save_every = max(50, min(1000, steps_per_epoch // 4)) if steps_per_epoch else 200
    logging_every = max(1, min(50, save_every // 10 or 1))
    eval_every = 0
    if payload.get("has_eval_data") and steps_per_epoch:
        eval_every = max(50, min(1000, steps_per_epoch // 2))
    workers = max(0, min(8, (os.cpu_count() or 4) - 1)) if not gpus else min(8, (os.cpu_count() or 4))
    learning_rate = {
        "lora": 2e-4,
        "qlora": 2e-4,
        "full_finetune": 2e-5,
        "sft": 1e-4,
        "continued_pretraining": 5e-5,
        "scratch": 3e-4,
    }.get(method, 2e-4)
    checkpoint_limit = 3 if not vram else max(2, min(5, int(vram / (4 * 1024**3)) or 2))

    suggested = {
        "method": method,
        "precision": precision,
        "quantization": quantization,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "learning_rate": learning_rate,
        "lr_scheduler": "cosine",
        "warmup_steps": max(0, min(100, steps_per_epoch // 20)) if steps_per_epoch else 20,
        "optimizer": "adamw",
        "save_every": save_every,
        "logging_every": logging_every,
        "eval_every": eval_every,
        "checkpoint_limit": checkpoint_limit,
        "gradient_checkpointing": bool(vram and model_params and vram < model_params * 4),
        "workers": workers,
        "seed": 42,
    }
    if suggested["gradient_checkpointing"]:
        reasons.append(
            "Gradient checkpointing is enabled: it trades roughly 20% speed for a large activation "
            "memory reduction, which this device needs.")
    if not gpus:
        reasons.append(
            "No CUDA device detected, so the run will use the CPU. With the tiny backend this is fully "
            "supported; the Transformers backend on CPU is only practical for very small models.")
    return {
        "labelled": "estimated",
        "suggested": suggested,
        "reasons": reasons,
        "hardware": {
            "gpus": [gpu["name"] for gpu in gpus],
            "vram": vram,
            "ram_total": ram,
            "ram_available": ram_free,
            "cpu_cores": os.cpu_count(),
        },
    }


def m_quantization_plan(payload: dict[str, Any]) -> dict[str, Any]:
    report = precision_tools.plan(
        payload.get("model") or payload.get("path"),
        str(payload.get("target_dtype") or payload.get("dtype") or "F16"),
        str(payload.get("target_format") or payload.get("format") or "safetensors"),
        payload.get("workspace"),
    )
    return report


def m_quantization_convert(payload: dict[str, Any]) -> dict[str, Any]:
    destination = payload.get("destination")
    if not destination:
        raise ZxError(code="bad_payload", message="A destination folder is required.",
                      hint="Pick an output folder outside the source model.")
    return precision_tools.convert(
        payload.get("model") or payload.get("path"),
        destination,
        str(payload.get("target_dtype") or payload.get("dtype") or "F16"),
        str(payload.get("target_format") or payload.get("format") or "safetensors"),
        payload.get("workspace"),
        progress=None,
    )


def m_adapters_scan(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain import adapters

    roots = [str(item) for item in payload.get("roots") or []]
    if not roots:
        workspace = payload.get("workspace")
        models_dir = payload.get("models_dir")
        if models_dir:
            roots = [str(models_dir)]
        elif workspace:
            roots = [str(workspace)]
    found = adapters.scan(roots, int(payload.get("max_depth") or 3))
    return {
        "roots": roots,
        "adapters": found,
        "count": len(found),
        "note": "Adapters are folders containing adapter_config.json; the config is read as written.",
    }


def m_adapters_describe(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain import adapters

    return adapters.describe(payload.get("path") or payload.get("adapter"))


def m_adapters_plan(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain import adapters

    return adapters.merge_plan(
        payload.get("path") or payload.get("adapter"),
        payload.get("base_model"),
        payload.get("destination"),
    )


def m_tools_report(payload: dict[str, Any]) -> dict[str, Any]:
    from zxtrain import tools

    return tools.report(payload.get("workspace"), payload.get("model"))


METHODS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "health": m_health,
    "engine.capabilities": m_engine_capabilities,
    "hardware.detect": m_hardware_detect,
    "hardware.live": m_hardware_live,
    "env.report": m_env_report,
    "env.plan": m_env_plan,
    "backends.list": m_backends_list,
    "models.scan": m_models_scan,
    "models.inspect": m_models_inspect,
    "models.import": m_models_import,
    "models.card": m_models_card,
    "datasets.inspect": m_datasets_inspect,
    "datasets.preview": m_datasets_preview,
    "datasets.mapping": m_datasets_mapping,
    "datasets.plan_clean": m_datasets_plan_clean,
    "datasets.clean": m_datasets_clean,
    "datasets.split": m_datasets_split,
    "datasets.export": m_datasets_export,
    "datasets.card": m_datasets_card,
    "datasets.from_conversations": m_datasets_from_conversations,
    "tokenizer.encode": m_tokenizer_encode,
    "tokenizer.decode": m_tokenizer_decode,
    "tokenizer.stats": m_tokenizer_stats,
    "training.prepare": m_training_prepare,
    "training.plan": m_training_plan,
    "training.auto_configure": m_auto_configure,
    "jobs.list": m_jobs_list,
    "jobs.summary": m_jobs_summary,
    "jobs.reconcile": m_jobs_reconcile,
    "jobs.logs": m_jobs_logs,
    "lineage.list": m_lineage_list,
    "storage.report": m_storage_report,
    "storage.tree": m_storage_tree,
    "storage.orphans": m_storage_orphans,
    "storage.clean": m_storage_clean,
    "storage.protect": m_storage_protect,
    "errors.diagnose": m_errors_diagnose,
    "reproducibility.bundle": m_reproducibility_bundle,
    "quantization.plan": m_quantization_plan,
    "quantization.convert": m_quantization_convert,
    "adapters.scan": m_adapters_scan,
    "adapters.describe": m_adapters_describe,
    "adapters.plan": m_adapters_plan,
    "tools.report": m_tools_report,
}


# --------------------------------------------------------------------------- #
# Sidecar (interactive inference)
# --------------------------------------------------------------------------- #

def serve() -> int:
    """Keep a model resident and answer requests as JSON lines.

    Request:  {"id": 1, "op": "load" | "generate" | "unload" | "status", ...}
    Response: {"id": 1, "event": {...}}  then  {"id": 1, "done": true, "result": {...}}
    """
    state: dict[str, Any] = {"model": None, "backend": None, "loaded_at": None}

    def respond(payload: dict[str, Any]) -> None:
        print(json.dumps(json_safe(payload), ensure_ascii=False), flush=True)

    respond({"ready": True, "engine": ENGINE_VERSION, "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            respond({"error": {"code": "bad_request", "message": "Invalid JSON line."}})
            continue
        request_id = request.get("id")
        op = request.get("op")
        try:
            if op == "load":
                path = str(request.get("model") or "")
                if not path or not Path(path).exists():
                    raise ZxError(code="model_missing", message=f"Model path does not exist: {path}",
                                  hint="Import or select a model first.")
                backend_id = request.get("backend") or ("tiny" if (Path(path) / "model.json").exists() else "hf")
                backend = registry.resolve(backend_id)
                if backend is None or not backend.info.available:
                    raise ZxError(
                        code="backend_unavailable",
                        message=f"{backend_id} backend is not available: "
                                f"{getattr(backend.info, 'reason', 'missing runtime')}",
                        hint="Install the ML runtime from the Environment page to load Transformers models.",
                    )
                if backend.info.id == "tiny":
                    from zxtrain.backends import tiny

                    model, tokenizer = tiny.load_model(path)
                    state.update({
                        "backend": "tiny", "model": path, "loaded_at": time.time(),
                        "parameters": model.parameter_count(),
                        "vocab_size": tokenizer.vocab_size,
                        "context_length": model.context_length,
                    })
                else:
                    state.update({"backend": "hf", "model": path, "loaded_at": time.time()})
                respond({"id": request_id, "done": True, "result": {
                    "model": path, "backend": state["backend"], "parameters": state.get("parameters"),
                    "context_length": state.get("context_length"),
                }})
            elif op == "status":
                respond({"id": request_id, "done": True, "result": {
                    "loaded": bool(state.get("model")), "model": state.get("model"),
                    "backend": state.get("backend"), "parameters": state.get("parameters"),
                }})
            elif op == "unload":
                state.update({"model": None, "backend": None})
                respond({"id": request_id, "done": True, "result": {"unloaded": True}})
            elif op == "generate":
                if not state.get("model"):
                    raise ZxError(code="no_model_loaded",
                                  message="No model is loaded in the playground.",
                                  hint="Load a model before sending a message.")
                backend = registry.resolve(state["backend"])
                result = backend.generate(state["model"], request, lambda event: respond(
                    {"id": request_id, "event": event}))
                respond({"id": request_id, "done": True, "result": result})
            elif op == "evaluate":
                backend = registry.resolve(request.get("backend") or state.get("backend") or "tiny")
                if backend is None or not backend.info.available:
                    raise ZxError(code="backend_unavailable", message="Selected backend is unavailable.")
                result = backend.evaluate(str(request.get("model") or state.get("model")),
                                          [str(path) for path in request.get("datasets") or []],
                                          request, lambda event: respond({"id": request_id, "event": event}))
                respond({"id": request_id, "done": True, "result": result})
            elif op == "ping":
                respond({"id": request_id, "done": True, "result": {"pong": True}})
            else:
                raise ZxError(code="unknown_op", message=f"Unknown operation: {op}",
                              hint="Use load, generate, evaluate, status or unload.")
        except ZxError as exc:
            respond({"id": request_id, "done": True, "error": exc.to_dict()})
        except BaseException as exc:  # noqa: BLE001
            respond({"id": request_id, "done": True, "error": from_exception(exc).to_dict()})
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(json.dumps({"ok": False, "error": {"code": "no_command",
                                                 "message": "No engine command given."}}))
        return 2
    command = argv[0]
    if command == "serve":
        return serve()
    if command == "run":
        if len(argv) < 2:
            print(json.dumps({"ok": False, "error": {"code": "no_spec",
                                                     "message": "run needs a spec file path."}}))
            return 2
        try:
            runner = trainer.JobRunner(Path(argv[1]))
            result = runner.run()
            print(json.dumps({"ok": result.get("status") != "failed", "data": result}, ensure_ascii=False))
            return 0 if result.get("status") != "failed" else 1
        except ZxError as exc:
            print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False))
            return 1
        except BaseException as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": from_exception(exc).to_dict()}, ensure_ascii=False))
            return 1
    if command == "install":
        # Environment installs run as jobs so the UI stays responsive.
        payload = _read_payload()
        events: list[dict[str, Any]] = []

        def emit(event: dict[str, Any]) -> None:
            events.append(event)
            print(f"@@event {json.dumps(json_safe({'ts': time.time(), **event}), ensure_ascii=False)}",
                  flush=True)

        try:
            result = environment.install_runtime(
                payload.get("workspace") or str(Path.home()),
                bool(payload.get("cuda")),
                payload.get("base_python"),
                payload.get("extra"),
                emit,
            )
            print(json.dumps({"ok": True, "data": result}, ensure_ascii=False))
            return 0
        except ZxError as exc:
            print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False))
            return 1

    handler = METHODS.get(command)
    if handler is None:
        print(json.dumps({"ok": False, "error": {
            "code": "unknown_command",
            "message": f"Unknown engine command: {command}",
            "hint": "Run with --help to list the available commands.",
        }}))
        return 2
    try:
        payload = _read_payload()
        data = handler(payload)
        print(json.dumps({"ok": True, "data": json_safe(data)}, ensure_ascii=False))
        return 0
    except ZxError as exc:
        print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False))
        return 1
    except BaseException as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": from_exception(exc).to_dict()}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    if "--help" in sys.argv:
        print("zxtrain engine\n\nCommands:")
        for name in sorted(METHODS):
            print(f"  {name}")
        print("  run <spec.json>")
        print("  install")
        print("  serve")
        raise SystemExit(0)
    raise SystemExit(main())
