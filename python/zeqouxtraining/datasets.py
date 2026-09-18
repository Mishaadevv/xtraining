"""Dataset loading, field auto-detection, normalisation and validation.

Everything in this module is real file I/O. Only the Hugging Face *datasets*
path needs an optional dependency; JSON/JSONL/CSV/TXT/folders work on bare
Python, which means the app can validate a dataset on any machine.

Normalisation is done by :func:`analyze`, which returns two parallel things:

* ``samples``  — only the usable samples, ready for training
* ``records``  — one status entry per *raw* record, so statistics never lose
  their alignment when a row in the middle is dropped

That distinction matters: an earlier index-based approach miscounted rows as
soon as one record was skipped.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".text", ".parquet"}
TEXT_EXTENSIONS = {".txt", ".text"}

CHAT_ROLES = {"system", "user", "assistant", "tool", "function"}

# Ordered by how common / unambiguous they are in real instruction datasets.
FIELD_PAIRS: list[tuple[str, ...]] = [
    ("instruction", "input", "output"),
    ("instruction", "output"),
    ("prompt", "completion"),
    ("prompt", "response"),
    ("question", "answer"),
    ("query", "response"),
    ("input", "output"),
    ("source", "target"),
    ("problem", "solution"),
]

TEXT_FIELDS: tuple[str, ...] = (
    "text", "content", "document", "body", "raw", "completion",
    "answer", "output", "response", "sentence", "paragraph",
)

MESSAGES_FIELDS: tuple[str, ...] = (
    "messages", "conversation", "conversations", "chat", "dialog", "dialogue", "thread",
)

LIST_CONTAINER_KEYS: tuple[str, ...] = ("data", "train", "records", "rows", "examples", "samples", "items")

TEMPLATES: dict[str, str] = {
    "alpaca": (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ),
    "plain": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "concise": "User: {prompt}\nAssistant: {response}",
}

SHORT_SAMPLE_CHARS = 3

ProgressFn = Callable[[str, int, int], None]


class DatasetError(Exception):
    """Raised for problems the user can act on (bad format, unreadable file)."""

    def __init__(self, message: str, *, hint: str = "", code: str = "dataset_error"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


# --------------------------------------------------------------------------- #
# Format detection and loading
# --------------------------------------------------------------------------- #

def detect_format(path: str) -> str:
    target = Path(path)
    if target.is_dir():
        return "folder"
    suffix = target.suffix.lower()
    if suffix == ".ndjson":
        return "jsonl"
    if suffix in (".txt", ".text"):
        return "txt"
    if suffix == ".tsv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    raise DatasetError(
        f"Unsupported file type '{suffix or target.name}'.",
        hint="Supported formats: .json, .jsonl, .csv, .tsv, .txt, .parquet and folders.",
        code="unsupported_format",
    )


def _read_text(path: Path, limit_bytes: int | None = None) -> str:
    try:
        with open(path, "rb") as handle:
            raw = handle.read(limit_bytes) if limit_bytes else handle.read()
    except OSError as exc:
        raise DatasetError(
            f"Could not read '{path.name}': {exc.strerror or exc}.",
            hint="Check that the file exists and is not locked by another program.",
            code="unreadable",
        ) from exc

    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _pick_list_from_object(obj: dict[str, Any]) -> tuple[list[Any] | None, str | None]:
    for key in LIST_CONTAINER_KEYS:
        value = obj.get(key)
        if isinstance(value, list):
            return value, key
    # Hugging Face split maps: {"train": [...], "validation": [...]}
    lists = {k: v for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)}
    if lists:
        key = "train" if "train" in lists else next(iter(lists))
        return lists[key], key
    return None, None


def load_records(
    path: str,
    *,
    fmt: str = "auto",
    max_records: int = 50_000,
    progress: ProgressFn | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Load up to ``max_records`` raw records plus metadata about the source."""
    target = Path(path)
    if not target.exists():
        raise DatasetError(
            f"'{path}' does not exist.",
            hint="Pick the dataset again — the file may have been moved or renamed.",
            code="not_found",
        )

    resolved = detect_format(path) if fmt in ("auto", "", None) else str(fmt)
    meta: dict[str, Any] = {
        "path": str(target.resolve()),
        "name": target.name,
        "format": resolved,
        "is_dir": target.is_dir(),
        "container_key": None,
        "files": [],
        "truncated": False,
    }

    if resolved == "folder":
        records = _load_folder(target, max_records, progress, meta)
    elif resolved in ("txt", "text"):
        records = _split_text(_read_text(target))
        meta["files"] = [target.name]
    elif resolved in ("json", "jsonl"):
        records = _load_json_like(target, resolved, max_records, meta)
    elif resolved == "csv":
        records = _load_csv(target, max_records)
        meta["files"] = [target.name]
    elif resolved == "parquet":
        records = _load_parquet(target, max_records)
        meta["files"] = [target.name]
    else:
        raise DatasetError(
            f"Unknown dataset format '{resolved}'.",
            hint="Supported formats: json, jsonl, csv, txt, parquet and folders.",
            code="unsupported_format",
        )

    if len(records) > max_records:
        records = records[:max_records]
        meta["truncated"] = True

    meta["bytes"] = _dir_size(target) if target.is_dir() else target.stat().st_size
    meta["records"] = len(records)
    return records, meta


def _dir_size(target: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(target):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _load_folder(
    target: Path,
    max_records: int,
    progress: ProgressFn | None,
    meta: dict[str, Any],
) -> list[Any]:
    candidates: list[Path] = []
    for root, _dirs, files in os.walk(target):
        for name in sorted(files):
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                candidates.append(Path(root) / name)

    if not candidates:
        raise DatasetError(
            f"No supported dataset files inside '{target.name}'.",
            hint="Folders may contain .json, .jsonl, .csv, .txt or .parquet files.",
            code="empty_folder",
        )

    meta["files"] = [str(p.relative_to(target)) for p in candidates]
    records: list[Any] = []
    file_errors: list[dict[str, Any]] = []
    shapes: dict[tuple[str, ...], list[str]] = {}

    for index, file_path in enumerate(candidates):
        if len(records) >= max_records:
            meta["truncated"] = True
            break
        if progress:
            progress(f"Reading {file_path.name}", index, len(candidates))
        remaining = max_records - len(records)
        try:
            sub, _sub_meta = load_records(str(file_path), max_records=remaining)
        except DatasetError as exc:
            # One bad file must not hide the good ones: report and continue.
            file_errors.append({
                "file": str(file_path.relative_to(target)).replace("\\", "/"),
                "message": exc.message,
                "code": exc.code,
            })
            continue

        # Track the record shape per file so a folder that mixes incompatible
        # datasets can be reported instead of silently mis-mapped.
        if sub and isinstance(sub[0], dict):
            signature = tuple(sorted(str(key).lower() for key in sub[0].keys()))
            if signature:
                shapes.setdefault(signature, []).append(
                    str(file_path.relative_to(target)).replace("\\", "/")
                )

        records.extend(sub)

    if len(shapes) > 1:
        meta["mixed_shapes"] = [
            {"fields": list(signature), "files": files}
            for signature, files in shapes.items()
        ]
    if file_errors:
        meta["file_errors"] = file_errors
    if not records:
        raise DatasetError(
            f"None of the {len(candidates)} files in '{target.name}' could be read.",
            hint=file_errors[0]["message"] if file_errors else
                 "Check that the folder contains valid dataset files.",
            code="folder_unreadable",
        )
    return records


def _load_json_like(
    target: Path,
    resolved: str,
    max_records: int,
    meta: dict[str, Any],
) -> list[Any]:
    text = _read_text(target)
    stripped = text.lstrip()

    if resolved == "jsonl" or (stripped and stripped[0] not in "[{"):
        return _parse_jsonl(text, max_records)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # A file named .json that is really JSONL is a very common case.
        if "\n" in text.strip():
            try:
                records = _parse_jsonl(text, max_records)
                meta["format"] = "jsonl"
                return records
            except DatasetError:
                pass
        raise DatasetError(
            f"'{target.name}' is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}.",
            hint="Fix the syntax error, or convert the file to JSONL (one JSON object per line).",
            code="invalid_json",
        ) from exc

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        picked, key = _pick_list_from_object(parsed)
        if picked is not None:
            meta["container_key"] = key
            return picked
        return [parsed]

    raise DatasetError(
        f"'{target.name}' contains a {type(parsed).__name__}, which cannot be used as a dataset.",
        hint="A dataset must be a JSON array of objects, or an object with a 'data' array.",
        code="invalid_shape",
    )


def _parse_jsonl(text: str, max_records: int) -> list[Any]:
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"Invalid JSON on line {line_number}: {exc.msg}.",
                hint="Every line of a JSONL file must be a complete JSON object.",
                code="invalid_jsonl",
            ) from exc
        if len(records) >= max_records:
            break
    if not records:
        raise DatasetError("The file contains no records.", code="empty")
    return records


def _load_csv(target: Path, max_records: int) -> list[Any]:
    text = _read_text(target)
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise DatasetError("The CSV file has no header row.", code="no_header")

    records: list[Any] = []
    for row in reader:
        records.append({k: (v if v is not None else "") for k, v in row.items() if k})
        if len(records) >= max_records:
            break
    if not records:
        raise DatasetError("The CSV file contains only a header.", code="empty")
    return records


def _load_parquet(target: Path, max_records: int) -> list[Any]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except Exception as exc:
        raise DatasetError(
            "Reading .parquet datasets requires the 'pyarrow' package.",
            hint="Install it with: pip install pyarrow",
            code="missing_dependency",
        ) from exc

    try:
        rows = pq.read_table(target).slice(0, max_records).to_pylist()
    except Exception as exc:
        raise DatasetError(
            f"Could not read '{target.name}' as parquet: {exc}",
            code="parquet_error",
        ) from exc
    if not rows:
        raise DatasetError("The parquet file contains no rows.", code="empty")
    return rows


def _split_text(text: str) -> list[Any]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise DatasetError("The file is empty.", code="empty")
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    if len(paragraphs) > 1:
        return [{"text": block} for block in paragraphs]
    return [{"text": line.strip()} for line in text.splitlines() if line.strip()]


def load_hf_dataset(dataset_id: str, split: str = "train", max_records: int = 50_000) -> list[Any]:
    """Load a Hugging Face dataset. Requires the optional 'datasets' package."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except Exception as exc:
        raise DatasetError(
            "Loading datasets from Hugging Face requires the 'datasets' package.",
            hint="Install it with: pip install datasets",
            code="missing_dependency",
        ) from exc

    try:
        dataset = load_dataset(dataset_id, split=split)
    except Exception as exc:
        raise DatasetError(
            f"Could not load '{dataset_id}' from Hugging Face: {exc}",
            hint="Check the dataset id, the split name, and your network connection.",
            code="hf_error",
        ) from exc

    subset = dataset.select(range(min(len(dataset), max_records)))
    return [dict(row) for row in subset]


def resolve_source(
    *,
    path: str | None = None,
    hf_id: str | None = None,
    split: str = "train",
    fmt: str = "auto",
    max_records: int = 20_000,
    progress: ProgressFn | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Load records from either a local path or a Hugging Face dataset id.

    Keeping both sources behind one function means validation, previewing and
    training all describe the dataset the same way.
    """
    if hf_id:
        records = load_hf_dataset(hf_id, split=split, max_records=max_records)
        return records, {
            "path": hf_id,
            "name": f"{hf_id} ({split})",
            "format": "hf",
            # No local file to measure; None keeps the report shape identical.
            "bytes": None,
            "records": len(records),
            "files": [],
            "source": "huggingface",
            "split": split,
            "truncated": len(records) >= max_records,
        }

    if not path:
        raise DatasetError(
            "No dataset path or Hugging Face dataset id was provided.",
            hint="Import a local file, or enter a dataset id such as `tatsu-lab/alpaca`.",
            code="no_source",
        )

    records, meta = load_records(path, fmt=fmt, max_records=max_records, progress=progress)
    meta["source"] = "local"
    return records, meta


# --------------------------------------------------------------------------- #
# Field auto-detection
# --------------------------------------------------------------------------- #

def _sample_keys(records: Iterable[Any], limit: int = 25) -> list[str]:
    """Union of keys across the first few records, in first-seen order.

    Datasets occasionally use a different field name for a single row (a
    ShareGPT export mixing ``messages`` and ``conversations``, for example).
    Sampling several records catches that instead of trusting row zero.
    """
    keys: list[str] = []
    for index, record in enumerate(records):
        if index >= limit:
            break
        if isinstance(record, dict):
            for key in record.keys():
                if key not in keys:
                    keys.append(key)
    return keys


def _looks_like_value(records: Iterable[Any], field: str, kinds: tuple[type, ...]) -> bool:
    for index, record in enumerate(records):
        if index >= 25:
            break
        if isinstance(record, dict) and isinstance(record.get(field), kinds):
            return True
    return False


def detect_mapping(records: list[Any]) -> dict[str, Any]:
    """Infer how to turn raw records into training samples.

    Returns a mapping the UI shows to the user, so an automatic decision is
    never hidden: ``{"kind": "chat"|"pair"|"text", ...}``.
    """
    keys = _sample_keys(records)
    if not keys:
        return {"kind": "text", "text_field": None, "auto": True}

    lowered = {key.lower(): key for key in keys}

    message_fields = [lowered[name] for name in MESSAGES_FIELDS if name in lowered]
    message_fields = [
        field for field in message_fields
        if _looks_like_value(records, field, (list, str))
    ]
    if message_fields:
        return {
            "kind": "chat",
            "messages_field": message_fields[0],
            "messages_field_alternatives": message_fields[1:],
            "fields_present": message_fields,
            "auto": True,
        }

    for pair in FIELD_PAIRS:
        if not all(part in lowered for part in pair):
            continue
        resolved = [lowered[part] for part in pair]
        if len(pair) == 3:
            return {
                "kind": "pair",
                "instruction_field": resolved[0],
                "input_field": resolved[1],
                "output_field": resolved[2],
                "template": "alpaca",
                "auto": True,
            }
        return {
            "kind": "pair",
            "instruction_field": resolved[0],
            "input_field": None,
            "output_field": resolved[1],
            "template": "plain",
            "auto": True,
        }

    for candidate in TEXT_FIELDS:
        if candidate in lowered:
            return {"kind": "text", "text_field": lowered[candidate], "auto": True}

    # Fall back to the longest string field, which is usually the content.
    string_fields = [
        key for key in keys
        if _looks_like_value(records, key, (str,)) and key.lower() != "id"
    ]
    if string_fields:
        def score(field: str) -> int:
            longest = 0
            for index, record in enumerate(records):
                if index >= 25 or not isinstance(record, dict):
                    break
                value = record.get(field)
                if isinstance(value, str):
                    longest = max(longest, len(value))
            return longest

        best = max(string_fields, key=score)
        return {"kind": "text", "text_field": best, "auto": True}

    return {"kind": "unknown", "auto": True, "fields": keys}


def _normalize_messages(value: Any) -> list[dict[str, str]] | None:
    """Accept OpenAI-style and ShareGPT-style message lists."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, list):
        return None

    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = item.get("role") or item.get("from") or item.get("sender")
        content = item.get("content")
        if content is None:
            content = item.get("value")
        if content is None:
            content = item.get("text")
        if role is None or content is None:
            return None
        role = str(role).lower()
        if role == "human":
            role = "user"
        elif role in ("gpt", "bot", "model"):
            role = "assistant"
        if role not in CHAT_ROLES:
            role = "user"
        messages.append({"role": role, "content": str(content)})
    return messages or None


def messages_to_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def render_pair(
    instruction: str,
    extra_input: str,
    output: str,
    template_text: str,
) -> str:
    if "{prompt}" in template_text:
        return template_text.format(prompt=instruction, response=output)
    if extra_input:
        return template_text.format(instruction=instruction, input=extra_input, output=output)
    return TEMPLATES["plain"].format(instruction=instruction, output=output)


def analyze(
    records: list[Any],
    mapping: dict[str, Any],
    template: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert raw records into samples plus a per-record status list.

    Returns ``(samples, statuses)`` where ``statuses`` has exactly one entry per
    raw record, so statistics stay aligned even when records are dropped.
    """
    kind = mapping.get("kind", "text")
    template_name = template or mapping.get("template") or "alpaca"
    template_text = TEMPLATES.get(template_name) or mapping.get("template_text") or TEMPLATES["plain"]

    samples: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    if kind == "chat":
        fields: list[str] = []
        primary = mapping.get("messages_field")
        if primary:
            fields.append(primary)
        fields.extend(mapping.get("messages_field_alternatives") or [])
        if not fields:
            fields = ["messages"]

        for record in records:
            messages = None
            if isinstance(record, dict):
                for field in fields:
                    messages = _normalize_messages(record.get(field))
                    if messages:
                        break
            elif isinstance(record, str):
                messages = _normalize_messages(record)

            if not messages:
                statuses.append({"usable": False, "reason": "no_messages",
                                 "text": "", "response_chars": 0})
                continue

            text = messages_to_text(messages)
            response_chars = len(messages[-1].get("content") or "")
            samples.append({"messages": messages})
            statuses.append({"usable": True, "reason": None,
                             "text": text, "response_chars": response_chars,
                             "roles": [m["role"] for m in messages]})
        return samples, statuses

    if kind == "pair":
        instruction_field = mapping.get("instruction_field")
        input_field = mapping.get("input_field")
        output_field = mapping.get("output_field")

        for record in records:
            if not isinstance(record, dict):
                statuses.append({"usable": False, "reason": "not_a_record",
                                 "text": "", "response_chars": 0})
                continue

            raw_instruction = record.get(instruction_field) if instruction_field else ""
            instruction = str(raw_instruction or "").strip()
            extra_input = str(record.get(input_field) or "").strip() if input_field else ""
            output = str(record.get(output_field) or "").strip() if output_field else ""

            # A record with no target has nothing to learn from, and a record
            # with no prompt is not a training pair either.
            if not output:
                statuses.append({"usable": False, "reason": "empty_output",
                                 "text": "", "response_chars": 0})
                continue
            if instruction_field and not instruction:
                statuses.append({"usable": False, "reason": "empty_prompt",
                                 "text": "", "response_chars": 0})
                continue

            text = render_pair(instruction, extra_input, output, template_text)
            samples.append({"text": text})
            statuses.append({"usable": True, "reason": None,
                             "text": text, "response_chars": len(output)})
        return samples, statuses

    # kind == "text", or an explicit text mapping.
    field = mapping.get("text_field")
    for record in records:
        text = ""
        if isinstance(record, str):
            text = record
        elif isinstance(record, dict):
            if field and isinstance(record.get(field), str):
                text = record[field]
            else:
                text = " ".join(
                    str(value) for value in record.values()
                    if isinstance(value, (str, int, float))
                )

        if not text.strip():
            statuses.append({"usable": False, "reason": "empty_text",
                             "text": "", "response_chars": 0})
            continue

        samples.append({"text": text})
        statuses.append({"usable": True, "reason": None,
                         "text": text, "response_chars": len(text)})
    return samples, statuses


def normalize(
    records: list[Any],
    mapping: dict[str, Any],
    template: str | None = None,
) -> list[dict[str, Any]]:
    """Only the usable samples — what the training backend consumes."""
    samples, _statuses = analyze(records, mapping, template)
    return samples


def sample_text(sample: dict[str, Any]) -> str:
    """Flatten a normalised sample back to text (for previews)."""
    if "text" in sample:
        return sample["text"]
    return messages_to_text(sample.get("messages", []))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """Character-based token estimate (~4 characters per token).

    Deliberately approximate: exact tokenisation needs the model's tokenizer,
    which is only resolved at training time. The report says so.
    """
    return max(1, round(len(text) / 4))


_REASON_TEXT = {
    "empty_output": "missing an output/target value",
    "empty_prompt": "missing a prompt/instruction value",
    "empty_text": "empty text",
    "no_messages": "no usable role/content messages",
    "not_a_record": "not a JSON object",
}


def _format_hint(fmt: str, path: str | None) -> str:
    """Best-effort format for a source that could not be read."""
    if fmt and fmt != "auto":
        return fmt
    if not path:
        return "unknown"
    try:
        return detect_format(path) or "unknown"
    except Exception:  # noqa: BLE001 - the format is cosmetic here
        suffix = Path(path).suffix.lstrip(".").lower()
        return suffix or "unknown"


def validate(
    path: str | None = None,
    *,
    hf_id: str | None = None,
    split: str = "train",
    fmt: str = "auto",
    mapping: dict[str, Any] | None = None,
    context_length: int = 512,
    max_records: int = 20_000,
    preview_limit: int = 8,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Validate a dataset and return a report the UI can render directly.

    Works for a local file, a folder of shards, or a Hugging Face dataset id.
    """
    label = hf_id or path or ""
    try:
        records, meta = resolve_source(
            path=path,
            hf_id=hf_id,
            split=split,
            fmt=fmt,
            max_records=max_records,
            progress=progress,
        )
    except DatasetError as exc:
        return {
            "ok": False,
            "status": "errors",
            "error": {"message": exc.message, "hint": exc.hint, "code": exc.code},
            "issues": [{
                "severity": "error",
                "code": exc.code,
                "message": exc.message,
                "hint": exc.hint,
            }],
            # The same keys the success path returns, so a failed validation
            # never makes the caller lose the format it already knew about.
            "dataset": {
                "path": label,
                "name": hf_id or (Path(label).name if label else ""),
                "format": "hf" if hf_id else _format_hint(fmt, label),
                "source": "huggingface" if hf_id else "local",
                "split": split if hf_id else None,
                "bytes": None,
                "records": 0,
                "files": [],
                "file_errors": [],
                "mixed_shapes": [],
                "container_key": None,
                "truncated": False,
            },
            "stats": {},
            "preview": [],
            "mapping": mapping or {"kind": "unknown"},
            "sampled": False,
        }

    resolved_mapping = mapping or detect_mapping(records)
    kind = resolved_mapping.get("kind", "unknown")
    issues: list[dict[str, Any]] = []

    for file_error in meta.get("file_errors", []):
        issues.append({
            "severity": "error",
            "code": file_error["code"],
            "message": f"{file_error['file']}: {file_error['message']}",
            "hint": "Fix or remove this file — the remaining files were still analysed.",
            "file": file_error["file"],
        })

    mixed = meta.get("mixed_shapes") or []
    if len(mixed) > 1:
        summary = "; ".join(
            f"{', '.join(shape['fields'][:5])} ({len(shape['files'])} file(s))"
            for shape in mixed
        )
        issues.append({
            "severity": "warning",
            "code": "mixed_folder",
            "message": f"This folder contains {len(mixed)} different record shapes: {summary}.",
            "hint": "A folder dataset works best when every file has the same fields. "
                    "Split the folder, or import the files one at a time and map fields manually.",
            "shapes": mixed,
        })

    if kind == "unknown":
        issues.append({
            "severity": "error",
            "code": "no_content_field",
            "message": "No instruction, prompt, text or messages field was found.",
            "hint": "Map the fields manually, or rename a column to instruction/output, "
                    "prompt/completion, text or messages.",
            "fields": resolved_mapping.get("fields", []),
        })

    samples, statuses = analyze(records, resolved_mapping) if kind != "unknown" else ([], [])

    total = len(records)
    usable = 0
    short = 0
    overlength = 0
    duplicate_count = 0
    empty_reasons: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    field_coverage: Counter[str] = Counter()
    reason_examples: dict[str, list[int]] = {}
    too_long_indices: list[int] = []
    short_indices: list[int] = []
    char_lengths: list[int] = []
    total_tokens = 0
    longest = 0
    longest_index = 0
    seen: set[str] = set()

    tracked_fields = [
        resolved_mapping.get("instruction_field"),
        resolved_mapping.get("input_field"),
        resolved_mapping.get("output_field"),
        resolved_mapping.get("text_field"),
        resolved_mapping.get("messages_field"),
        *(resolved_mapping.get("messages_field_alternatives") or []),
    ]
    tracked_fields = [field for field in tracked_fields if field]

    for index, record in enumerate(records):
        if isinstance(record, dict) and tracked_fields:
            for field in tracked_fields:
                if record.get(field) not in (None, "", [], {}):
                    field_coverage[field] += 1

        status = statuses[index] if index < len(statuses) else {"usable": False, "reason": "not_a_record"}
        if not status["usable"]:
            reason = status.get("reason") or "unknown"
            empty_reasons[reason] += 1
            reason_examples.setdefault(reason, []).append(index)
            continue

        usable += 1
        text = status["text"]
        length = len(text.strip())
        char_lengths.append(length)
        if length > longest:
            longest = length
            longest_index = index

        tokens = estimate_tokens(text)
        total_tokens += tokens
        if tokens > context_length:
            overlength += 1
            if len(too_long_indices) < 20:
                too_long_indices.append(index)

        response_chars = status.get("response_chars") or length
        if response_chars < SHORT_SAMPLE_CHARS:
            short += 1
            if len(short_indices) < 20:
                short_indices.append(index)

        for role in status.get("roles") or []:
            role_counter[str(role)] += 1

        fingerprint = hashlib.blake2b(text.strip().encode("utf-8", "replace"),
                                      digest_size=16).hexdigest()
        if fingerprint in seen:
            duplicate_count += 1
        else:
            seen.add(fingerprint)

    # ------------------------------------------------------------------ issues
    if total == 0:
        issues.append({
            "severity": "error",
            "code": "no_records",
            "message": "The dataset contains no records.",
            "hint": "Check that the file is not empty and uses the format you selected.",
        })

    empty_total = sum(empty_reasons.values())
    if empty_total:
        breakdown = ", ".join(
            f"{count:,} {_REASON_TEXT.get(reason, reason)}"
            for reason, count in empty_reasons.most_common()
        )
        issues.append({
            "severity": "error" if empty_total >= total else "warning",
            "code": "unusable_records",
            "message": f"{empty_total:,} of {total:,} records cannot be used: {breakdown}.",
            "hint": "Fill in the missing fields or remove these rows before training.",
            "count": empty_total,
            "breakdown": dict(empty_reasons),
            "samples": sorted({i for indices in reason_examples.values() for i in indices})[:20],
        })

    if short:
        issues.append({
            "severity": "warning",
            "code": "very_short",
            "message": f"{short:,} records have a target shorter than {SHORT_SAMPLE_CHARS} characters.",
            "hint": "Very short targets teach the model to answer with almost nothing.",
            "count": short,
            "samples": short_indices,
        })

    if duplicate_count:
        issues.append({
            "severity": "warning",
            "code": "duplicates",
            "message": f"{duplicate_count:,} duplicate records detected "
                       f"({round(100 * duplicate_count / max(1, usable), 1)}% of usable rows).",
            "hint": "Duplicates bias training towards repeated content. Consider deduplicating.",
            "count": duplicate_count,
        })

    if overlength:
        issues.append({
            "severity": "warning",
            "code": "over_length",
            "message": f"{overlength:,} records exceed the {context_length}-token context length "
                       f"and will be truncated.",
            "hint": f"Raise Context length, or shorten those samples. Longest sample is record "
                    f"#{longest_index} at about {estimate_tokens('x' * longest):,} tokens.",
            "count": overlength,
            "samples": too_long_indices,
        })

    if kind != "unknown" and usable > 0 and usable == duplicate_count + 1 and usable > 2:
        issues.append({
            "severity": "warning",
            "code": "near_constant",
            "message": "Almost every record is a duplicate of the same text.",
            "hint": "This usually means the field mapping picked the wrong column.",
        })

    average_length = round(sum(char_lengths) / len(char_lengths), 1) if char_lengths else 0
    has_error = any(issue["severity"] == "error" for issue in issues)
    status = "errors" if has_error else ("warnings" if issues else "ok")

    preview = [sample_text(sample)[:1200] for sample in samples[:preview_limit]]

    return {
        "ok": not has_error and usable > 0,
        "status": status,
        "error": None,
        "issues": issues,
        "mapping": resolved_mapping,
        "dataset": {
            "path": meta["path"],
            "name": meta["name"],
            "format": meta["format"],
            "source": meta.get("source"),
            "split": meta.get("split"),
            "bytes": meta.get("bytes"),
            "records": total,
            "files": meta.get("files", []),
            "file_errors": meta.get("file_errors", []),
            "mixed_shapes": meta.get("mixed_shapes", []),
            "container_key": meta.get("container_key"),
            "truncated": meta.get("truncated", False),
        },
        "stats": {
            "records": total,
            "usable": usable,
            "empty": empty_total,
            "empty_reasons": dict(empty_reasons),
            "short": short,
            "duplicates": duplicate_count,
            "over_length": overlength,
            "avg_chars": average_length,
            "max_chars": longest,
            "max_chars_index": longest_index,
            "est_tokens_total": total_tokens,
            "est_tokens_avg": round(total_tokens / max(1, usable)),
            "roles": dict(role_counter),
            "field_coverage": {
                field: round(100 * count / max(1, total), 1)
                for field, count in field_coverage.items()
            },
        },
        "preview": preview,
        "sampled": total >= max_records,
        "context_length": context_length,
    }
