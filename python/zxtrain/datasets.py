"""Dataset reading, validation, cleaning, splitting and export.

Readers use the standard library by default (JSON/JSONL/CSV/TSV/TXT/Markdown/
SQLite/compressed text). Parquet, Arrow and Excel are read through PyArrow when
it is installed; when it is not, the engine says exactly which package is
missing instead of silently pretending the dataset is empty.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import itertools
import json
import lzma
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ZxError
from .util import (
    dir_size,
    ensure_dir,
    human_bytes,
    json_safe,
    now_iso,
    read_json,
    safe_slug,
    write_json_atomic,
)

STRUCTURED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".psv", ".parquet",
                       ".arrow", ".feather", ".orc", ".db", ".sqlite", ".sqlite3",
                       ".xlsx", ".xls", ".yaml", ".yml"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".log"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

ROLE_ALIASES: dict[str, list[str]] = {
    "system": ["system", "sys", "system_prompt"],
    "user": ["user", "prompt", "instruction", "question", "input", "query", "human"],
    "assistant": ["assistant", "response", "output", "answer", "completion", "chosen", "gpt", "target"],
    "rejected": ["rejected", "negative", "bad", "worse"],
    "label": ["label", "class", "category", "target_label"],
    "text": ["text", "content", "body", "document", "raw", "corpus"],
}

MAX_SCAN_BYTES = 512 * 1024 * 1024


def _open_text(path: Path) -> io.TextIOBase:
    suffix = path.suffix.lower()
    try:
        if suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if suffix == ".bz2":
            return bz2.open(path, "rt", encoding="utf-8", errors="replace")
        if suffix in (".xz", ".lzma"):
            return lzma.open(path, "rt", encoding="utf-8", errors="replace")
        return path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ZxError(
            code="dataset_unreadable",
            message=f"Cannot read {path.name}: {exc}",
            hint="Check that the file is not locked by another program and that its drive is connected.",
        ) from exc


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def _record_from_line(line: str) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        return {}
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return {"text": stripped}
    if isinstance(value, dict):
        return _normalise(value)
    if isinstance(value, list):
        return {"messages": _normalise(value)}
    return {"text": value}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line in handle:
            record = _record_from_line(line)
            if record:
                yield record


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    text = _open_text(path).read(MAX_SCAN_BYTES)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Fall back to JSONL: many "json" files in the wild are line-delimited.
        fallback = list(_iter_jsonl(path))
        if fallback:
            return iter(fallback)
        raise ZxError(
            code="dataset_invalid",
            message=f"{path.name} is not valid JSON: {exc.msg} (line {exc.lineno})",
            hint="Re-export the file, or import it as JSONL if it is line delimited.",
        ) from exc
    if isinstance(payload, list):
        return iter([_normalise(item) if isinstance(item, dict) else {"text": item} for item in payload])
    if isinstance(payload, dict):
        for key in ("data", "rows", "records", "examples", "samples", "conversations", "items"):
            if isinstance(payload.get(key), list):
                return iter([
                    _normalise(item) if isinstance(item, dict) else {"text": item}
                    for item in payload[key]
                ])
        return iter([_normalise(payload)])
    return iter([{"text": payload}])


def _iter_csv(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = {".tsv": "\t", ".psv": "|"}.get(path.suffix.lower())
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            yield {str(k): v for k, v in row.items() if k is not None}


def _iter_plain(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield {"text": text}


def _iter_sqlite(path: Path) -> Iterator[dict[str, Any]]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ZxError(code="dataset_invalid", message=f"Cannot open {path.name}: {exc}") from exc
    try:
        cursor = connection.cursor()
        tables = [row[0] for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        for table in tables:
            columns = [row[1] for row in cursor.execute(f'PRAGMA table_info("{table}")').fetchall()]
            for row in cursor.execute(f'SELECT * FROM "{table}"'):
                yield {name: value for name, value in zip(columns, row)}
    finally:
        connection.close()


def _iter_yaml(path: Path) -> Iterator[dict[str, Any]]:
    if _module("yaml") is None:
        raise ZxError(
            code="missing_dependency",
            message="YAML datasets need the PyYAML package.",
            hint="Install it from the Environment page, or export the data as JSONL.",
        )
    import yaml  # type: ignore

    payload = yaml.safe_load(_open_text(path).read())
    if isinstance(payload, list):
        for item in payload:
            yield _normalise(item) if isinstance(item, dict) else {"text": item}
    elif isinstance(payload, dict):
        yield _normalise(payload)


def _module(name: str):
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    try:
        return __import__(name)
    except Exception:
        return None


def _iter_pyarrow(path: Path, batch_size: int = 512) -> Iterator[dict[str, Any]]:
    if _module("pyarrow") is None:
        raise ZxError(
            code="missing_dependency",
            message=f"{path.suffix} datasets need the pyarrow package.",
            hint="Install pyarrow from the Environment page, or export the data as JSONL/CSV.",
        )
    import pyarrow.parquet as pq  # type: ignore

    if path.suffix.lower() == ".parquet":
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield _normalise(row)
        return
    import pyarrow.feather as feather  # type: ignore

    table = feather.read_table(path)
    for row in table.to_pylist():
        yield _normalise(row)


def _iter_excel(path: Path) -> Iterator[dict[str, Any]]:
    module = _module("openpyxl")
    if module is None:
        raise ZxError(
            code="missing_dependency",
            message="Excel datasets need the openpyxl package.",
            hint="Install openpyxl from the Environment page, or export the sheet as CSV.",
        )
    workbook = module.load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        try:
            header = [str(cell) if cell is not None else f"column_{index}" for index, cell in enumerate(next(rows))]
        except StopIteration:
            continue
        for row in rows:
            yield {name: value for name, value in zip(header, row)}


def iter_records(path: Path, limit: int | None = None, offset: int = 0) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        yield from _iter_directory(path, limit=limit, offset=offset)
        return
    suffix = path.suffix.lower()
    if suffix in (".gz", ".bz2", ".xz", ".lzma"):
        suffix = Path(path.stem).suffix.lower() or ".txt"
    producers = {
        ".jsonl": _iter_jsonl, ".ndjson": _iter_jsonl,
        ".json": _iter_json,
        ".csv": _iter_csv, ".tsv": _iter_csv, ".psv": _iter_csv,
        ".txt": _iter_plain, ".md": _iter_plain, ".markdown": _iter_plain,
        ".text": _iter_plain, ".log": _iter_plain,
        ".db": _iter_sqlite, ".sqlite": _iter_sqlite, ".sqlite3": _iter_sqlite,
        ".parquet": _iter_pyarrow, ".arrow": _iter_pyarrow, ".feather": _iter_pyarrow,
        ".xlsx": _iter_excel, ".xls": _iter_excel,
        ".yaml": _iter_yaml, ".yml": _iter_yaml,
    }
    producer = producers.get(suffix)
    if producer is None:
        raise ZxError(
            code="unsupported_format",
            message=f"Unsupported dataset format: {path.suffix or 'no extension'}",
            hint="Supported: JSONL, JSON, CSV, TSV, PSV, TXT, Markdown, Parquet, Arrow, SQLite, Excel, YAML.",
        )
    emitted = 0
    skipped = 0
    for record in producer(path):
        if not record:
            continue
        if skipped < offset:
            skipped += 1
            continue
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def _iter_directory(root: Path, limit: int | None = None, offset: int = 0) -> Iterator[dict[str, Any]]:
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    data_files = [p for p in files if _is_data_file(p)]
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    audio = [p for p in files if p.suffix.lower() in AUDIO_SUFFIXES]
    if data_files:
        emitted = 0
        skipped = 0
        for path in data_files:
            for record in iter_records(path):
                if skipped < offset:
                    skipped += 1
                    continue
                record = dict(record)
                record.setdefault("__source", path.relative_to(root).as_posix())
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
        return
    if images or audio:
        media = images + audio
        for index, path in enumerate(media):
            if index < offset:
                continue
            yield {
                "__media": path.relative_to(root).as_posix(),
                "__kind": "image" if path.suffix.lower() in IMAGE_SUFFIXES else "audio",
                "text": "",
            }
            if limit is not None and index - offset + 1 >= limit:
                return
        return
    raise ZxError(
        code="dataset_empty",
        message=f"No readable data files found under {root}",
        hint="A folder dataset must contain JSONL/JSON/CSV/TXT/Parquet shards, images or audio.",
    )


def _is_data_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in (".gz", ".bz2", ".xz", ".lzma"):
        suffix = Path(path.stem).suffix.lower()
    return suffix in STRUCTURED_SUFFIXES or suffix in TEXT_SUFFIXES


def count_records(path: Path, max_count: int = 5_000_000) -> int:
    total = 0
    for _ in iter_records(path):
        total += 1
        if total >= max_count:
            break
    return total


def detect_mapping(fields: Iterable[str], sample: dict[str, Any] | None = None) -> dict[str, Any]:
    """Guess how raw fields map onto training roles, with confidence."""
    fields = list(fields)
    lowered = {field.lower(): field for field in fields}
    mapping: dict[str, Any] = {}
    confidence: dict[str, str] = {}
    used: set[str] = set()

    for role, aliases in ROLE_ALIASES.items():
        for alias in aliases:
            if alias in lowered and lowered[alias] not in used:
                mapping[role] = lowered[alias]
                confidence[role] = "exact"
                used.add(lowered[alias])
                break
    # Conversation-style datasets: a list of {role, content} objects.
    if sample:
        for key, value in sample.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                keys = {str(k).lower() for k in value[0]}
                if {"role", "content"} <= keys or {"from", "value"} <= keys:
                    mapping["messages"] = key
                    confidence["messages"] = "conversation structure detected"
                    break
    if "text" not in mapping and not mapping.get("messages"):
        for field in fields:
            if lowered.get("text") == field:
                mapping["text"] = field
                confidence["text"] = "exact"
                break
    return {
        "mapping": mapping,
        "confidence": confidence,
        "fields": fields,
        "unmapped": [field for field in fields if field not in used and field not in mapping.values()],
    }


def normalise_record(record: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    """Turn one raw record into the canonical {'messages': [...]} or {'text': ...} shape."""
    if mapping.get("messages") and isinstance(record.get(mapping["messages"]), list):
        raw = record[mapping["messages"]]
        messages: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("from") or "").strip().lower()
            content = item.get("content", item.get("value", ""))
            role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role, role)
            if role and content not in (None, ""):
                messages.append({"role": role, "content": str(content)})
        if messages:
            return {"messages": messages}

    system = record.get(mapping["system"]) if mapping.get("system") else None
    user = record.get(mapping["user"]) if mapping.get("user") else None
    assistant = record.get(mapping["assistant"]) if mapping.get("assistant") else None
    if user is not None and assistant is not None:
        messages = []
        if system not in (None, ""):
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(user)})
        messages.append({"role": "assistant", "content": str(assistant)})
        return {"messages": messages}

    text = record.get(mapping.get("text", "")) if mapping.get("text") else None
    if text in (None, "") and mapping.get("text"):
        text = record.get(mapping["text"])
    if text not in (None, ""):
        return {"text": str(text)}
    return {}


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
    "{% endfor %}"
    "<|assistant|>\n"
)


def record_to_text(record: dict[str, Any], mapping: dict[str, Any], template: str = "chatml") -> str:
    normalised = normalise_record(record, mapping)
    if "messages" in normalised:
        parts: list[str] = []
        for message in normalised["messages"]:
            tag = message["role"]
            if template == "chatml":
                parts.append(f"<|{tag}|>\n{message['content']}")
            elif template == "plain":
                parts.append(f"{tag}: {message['content']}")
            else:
                parts.append(f"### {tag.capitalize()}:\n{message['content']}")
        if template == "chatml":
            parts.append("<|assistant|>\n")
        return "\n".join(parts)
    return str(normalised.get("text", ""))


def extract_texts(
    path: Path,
    mapping: dict[str, Any],
    limit: int | None = None,
    template: str = "chatml",
    min_chars: int = 1,
) -> list[str]:
    texts: list[str] = []
    for record in iter_records(path, limit=None if limit is None else limit * 3):
        text = record_to_text(record, mapping, template)
        if text and len(text) >= min_chars:
            texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return texts


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _percentiles(values: list[int], points: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, int | None]:
    if not values:
        return {f"p{int(point * 100)}": None for point in points}
    ordered = sorted(values)
    result: dict[str, int | None] = {}
    for point in points:
        index = min(len(ordered) - 1, max(0, int(round(point * (len(ordered) - 1)))))
        result[f"p{int(point * 100)}"] = ordered[index]
    return result


def _record_length(record: dict[str, Any], mapping: dict[str, Any]) -> int:
    text = record_to_text(record, mapping, "plain")
    return len(text)


def _record_digest(record: dict[str, Any], mapping: dict[str, Any]) -> str:
    payload = json.dumps(normalise_record(record, mapping) or record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _shingles(text: str, size: int = 5) -> set[str]:
    normalised = re.sub(r"\s+", " ", text.lower())
    if len(normalised) <= size:
        return {normalised}
    return {normalised[i:i + size] for i in range(len(normalised) - size + 1)}


def inspect_dataset(
    path: str | Path,
    mapping: dict[str, Any] | None = None,
    sample_size: int = 4000,
    near_duplicate_threshold: float = 0.9,
) -> dict[str, Any]:
    """Full structural + quality report over a real dataset."""
    root = Path(path).expanduser()
    if not root.exists():
        raise ZxError(
            code="dataset_missing",
            message=f"Dataset path does not exist: {root}",
            hint="Re-import the dataset or check that the drive is mounted.",
        )
    field_types: dict[str, Counter] = {}
    missing: Counter = Counter()
    lengths: list[int] = []
    digests: dict[str, int] = {}
    duplicates = 0
    empty = 0
    total = 0
    sample: dict[str, Any] | None = None
    preview: list[dict[str, Any]] = []
    signature: dict[str, int] = {}
    shingles_seen: list[set[str]] = []
    near_duplicates = 0
    media_kind = None

    # Detect the field mapping from the first records when the caller did not
    # supply one, so every statistic below measures the text users will train on.
    head: list[dict[str, Any]] = []
    for record in iter_records(root):
        head.append(record)
        if len(head) >= 50:
            break
    if not head:
        return _empty_report(root)
    head_fields: list[str] = []
    for record in head:
        for key in record:
            if key not in head_fields:
                head_fields.append(key)
    resolved_mapping: dict[str, Any] = dict(mapping) if mapping else {}
    if not resolved_mapping:
        resolved_mapping = detect_mapping(head_fields, head[0])["mapping"]
    mapping = resolved_mapping

    for record in itertools.chain(head, iter_records(root, offset=len(head))):
        if sample is None:
            sample = record
        total += 1
        if "__media" in record:
            media_kind = record.get("__kind")
        for field, value in record.items():
            field_types.setdefault(field, Counter())[_type_of(value)] += 1
            if value in (None, "", [], {}):
                missing[field] += 1
        if total <= sample_size:
            length = _record_length(record, mapping or {})
            lengths.append(length)
            if length == 0 or not normalise_record(record, mapping or {}):
                empty += 1
            digest = _record_digest(record, mapping or {})
            digests[digest] = digests.get(digest, 0) + 1
            if digests[digest] > 1:
                duplicates += 1
            if near_duplicate_threshold < 1 and length > 20:
                current = _shingles(record_to_text(record, mapping or {}, "plain"))
                signature[len(current)] = signature.get(len(current), 0) + 1
                for other in shingles_seen[-400:]:
                    if not current or not other:
                        continue
                    overlap = len(current & other) / max(1, len(current | other))
                    if overlap >= near_duplicate_threshold:
                        near_duplicates += 1
                        break
                shingles_seen.append(current)
        if len(preview) < 12:
            preview.append({
                "index": total - 1,
                "record": json_safe(record),
                "normalised": json_safe(normalise_record(record, mapping or {})),
                "length": _record_length(record, mapping or {}),
            })

    fields = list(field_types.keys())
    detected = mapping or detect_mapping(fields, sample)["mapping"]
    size_bytes, file_count = dir_size(root)
    token_estimate = sum(lengths) // 4 if lengths else None
    return {
        "path": str(root),
        "name": root.name,
        "kind": "directory" if root.is_dir() else root.suffix.lower().lstrip("."),
        "size_bytes": size_bytes,
        "size_human": human_bytes(size_bytes),
        "file_count": file_count,
        "record_count": total,
        "sampled": min(total, sample_size),
        "fields": [
            {
                "name": field,
                "types": dict(field_types[field].most_common(3)),
                "type": field_types[field].most_common(1)[0][0],
                "missing": missing[field],
                "missing_percent": round(missing[field] / total * 100, 2) if total else None,
            }
            for field in fields
        ],
        "field_names": fields,
        "duplicates": duplicates,
        "duplicate_percent": round(duplicates / max(1, len(lengths)) * 100, 2) if lengths else None,
        "near_duplicates": near_duplicates if near_duplicate_threshold < 1 else None,
        "near_duplicate_threshold": near_duplicate_threshold if near_duplicate_threshold < 1 else None,
        "empty_records": empty,
        "length": {
            "average": round(sum(lengths) / len(lengths), 1) if lengths else None,
            "max": max(lengths) if lengths else None,
            "min": min(lengths) if lengths else None,
            **_percentiles(lengths),
            "histogram": _histogram(lengths),
        },
        "token_estimate": {
            "labelled": "estimated",
            "chars_per_token": 4,
            "total": token_estimate,
            "average": round(token_estimate / len(lengths), 1) if (lengths and token_estimate) else None,
            "note": "Heuristic estimate (characters / 4). Train a tokenizer for an exact count.",
        },
        "media_kind": media_kind,
        "detected_mapping": detected,
        "mapping_suggestion": detect_mapping(fields, sample),
        "preview": preview,
        "inspected_at": now_iso(),
    }


def _empty_report(root: Path) -> dict[str, Any]:
    size_bytes, file_count = dir_size(root)
    return {
        "path": str(root),
        "name": root.name,
        "kind": "directory" if root.is_dir() else root.suffix.lower().lstrip("."),
        "size_bytes": size_bytes,
        "size_human": human_bytes(size_bytes),
        "file_count": file_count,
        "record_count": 0,
        "sampled": 0,
        "fields": [],
        "field_names": [],
        "duplicates": 0,
        "duplicate_percent": None,
        "near_duplicates": None,
        "empty_records": 0,
        "length": {"average": None, "max": None, "min": None, "histogram": []},
        "token_estimate": {"labelled": "estimated", "total": 0, "average": None,
                           "note": "No records were found in this dataset."},
        "media_kind": None,
        "detected_mapping": {},
        "mapping_suggestion": {"mapping": {}, "confidence": {}, "fields": [], "unmapped": []},
        "preview": [],
        "inspected_at": now_iso(),
        "empty": True,
    }


def _histogram(lengths: list[int], buckets: int = 12) -> list[dict[str, Any]]:
    if not lengths:
        return []
    maximum = max(lengths) or 1
    width = max(1, maximum // buckets)
    counts = Counter(min(buckets - 1, value // width) for value in lengths)
    return [
        {"from": index * width, "to": (index + 1) * width, "count": counts.get(index, 0)}
        for index in range(buckets)
    ]


# --------------------------------------------------------------------------- #
# Cleaning / transformation
# --------------------------------------------------------------------------- #

def plan_cleaning(path: Path, mapping: dict[str, Any], operations: list[dict[str, Any]], preview_limit: int = 8) -> dict[str, Any]:
    """Apply operations in memory and report what would change, before writing."""
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    stats: dict[str, int] = {op["type"]: 0 for op in operations}
    kept = 0
    for record in iter_records(path):
        before.append(record)
        current: dict[str, Any] | None = record
        for operation in operations:
            current, changed, reason = _apply_operation(current, mapping, operation)
            if changed:
                stats[operation["type"]] = stats.get(operation["type"], 0) + 1
            if current is None:
                stats.setdefault(f"{operation['type']}_dropped", 0)
                stats[f"{operation['type']}_dropped"] += 1
                break
        if current is not None:
            kept += 1
            after.append(current)
    return {
        "input_records": len(before),
        "output_records": kept,
        "dropped": len(before) - kept,
        "operation_stats": stats,
        "preview": [
            {
                "before": json_safe(before[index]) if index < len(before) else None,
                "after": json_safe(after[index]) if index < len(after) else None,
            }
            for index in range(min(preview_limit, max(len(before), len(after))))
        ],
    }


def _apply_operation(
    record: dict[str, Any] | None,
    mapping: dict[str, Any],
    operation: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool, str]:
    if record is None:
        return None, False, ""
    kind = operation.get("type")
    if kind == "drop_empty":
        normalised = normalise_record(record, mapping)
        if not normalised or (normalised.get("text") == "" and not normalised.get("messages")):
            return None, True, "empty"
        return record, False, ""
    if kind == "trim_whitespace":
        changed = False
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, str):
                new_value = re.sub(r"[ \t]+", " ", value).strip()
                changed = changed or new_value != value
                cleaned[key] = new_value
            else:
                cleaned[key] = value
        return cleaned, changed, "whitespace normalised"
    if kind == "strip_html":
        changed = False
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, str) and re.search(r"<[a-zA-Z/][^>]*>", value):
                new_value = re.sub(r"<[^>]+>", " ", value)
                new_value = re.sub(r"\s+", " ", new_value).strip()
                changed = True
                cleaned[key] = new_value
            else:
                cleaned[key] = value
        return cleaned, changed, "html tags removed"
    if kind == "length_filter":
        length = _record_length(record, mapping)
        minimum = int(operation.get("min", 0) or 0)
        maximum = int(operation.get("max", 0) or 0)
        if maximum and length > maximum:
            return None, True, "above max length"
        if length < minimum:
            return None, True, "below min length"
        return record, False, ""
    if kind == "drop_duplicates":
        return record, False, ""  # handled at the dataset level by the caller
    if kind == "regex_replace":
        pattern = operation.get("pattern")
        replacement = str(operation.get("replacement", ""))
        field = operation.get("field")
        if not pattern:
            return record, False, ""
        try:
            compiled = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            raise ZxError(
                code="invalid_regex",
                message=f"Invalid regular expression: {exc}",
                hint="Check the pattern; Python syntax is expected (for example \\s+ → ' ').",
            ) from exc
        changed = False
        cleaned = dict(record)
        for key in ([field] if field else list(record.keys())):
            value = cleaned.get(key)
            if isinstance(value, str):
                new_value = compiled.sub(replacement, value)
                if new_value != value:
                    changed = True
                    cleaned[key] = new_value
        return cleaned, changed, "regex applied"
    if kind == "field_select":
        keep = set(operation.get("fields") or [])
        if not keep:
            return record, False, ""
        cleaned = {key: value for key, value in record.items() if key in keep}
        return cleaned, len(cleaned) != len(record), "fields reduced"
    if kind == "rename_field":
        source = operation.get("from")
        target = operation.get("to")
        if not source or not target or source not in record:
            return record, False, ""
        cleaned = {target if key == source else key: value for key, value in record.items()}
        return cleaned, True, f"{source} → {target}"
    return record, False, ""


def write_records(records: Iterable[dict[str, Any]], destination: Path, fmt: str = "jsonl") -> Path:
    destination = Path(destination)
    ensure_dir(destination.parent)
    fmt = fmt.lower()
    if fmt in ("jsonl", "ndjson"):
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
    elif fmt == "json":
        payload = [json_safe(record) for record in records]
        write_json_atomic(destination, payload)
    elif fmt in ("csv", "tsv"):
        delimiter = "\t" if fmt == "tsv" else ","
        iterator = iter(records)
        try:
            first = next(iterator)
        except StopIteration:
            destination.write_text("", encoding="utf-8")
            return destination
        fields = list(first.keys())
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerow({key: _csv_value(first.get(key)) for key in fields})
            for record in iterator:
                writer.writerow({key: _csv_value(record.get(key)) for key in fields})
    elif fmt in ("txt", "text", "md"):
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                if "text" in record:
                    handle.write(str(record["text"]) + "\n")
                else:
                    handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
    elif fmt == "parquet":
        if _module("pyarrow") is None:
            raise ZxError(
                code="missing_dependency",
                message="Parquet export needs the pyarrow package.",
                hint="Install pyarrow from the Environment page, or export as JSONL.",
            )
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        rows = [json_safe(record) for record in records]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, destination)
    else:
        raise ZxError(
            code="unsupported_format",
            message=f"Cannot export as {fmt}",
            hint="Use jsonl, json, csv, tsv, txt or parquet.",
        )
    return destination


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_safe(value), ensure_ascii=False)
    return value


def clean_dataset(
    path: Path,
    mapping: dict[str, Any],
    operations: list[dict[str, Any]],
    destination: Path,
    fmt: str = "jsonl",
) -> dict[str, Any]:
    """Run the real cleaning pass and write the output plus an audit entry."""
    seen: set[str] = set()
    stats: Counter = Counter()
    kept = 0
    total = 0
    dedupe = any(operation.get("type") == "drop_duplicates" for operation in operations)
    near_threshold = next(
        (float(op.get("threshold", 0.9)) for op in operations if op.get("type") == "near_duplicates"),
        None,
    )
    shingles: list[set[str]] = []

    def generator() -> Iterator[dict[str, Any]]:
        nonlocal kept, total
        for record in iter_records(path):
            total += 1
            original = record
            current: dict[str, Any] | None = record
            for operation in operations:
                current, changed, _reason = _apply_operation(current, mapping, operation)
                if changed:
                    stats[operation["type"]] += 1
                if current is None:
                    stats[f"{operation['type']}_dropped"] += 1
                    break
            if current is None:
                continue
            if dedupe:
                digest = _record_digest(current, mapping)
                if digest in seen:
                    stats["drop_duplicates_dropped"] += 1
                    continue
                seen.add(digest)
            if near_threshold:
                text = record_to_text(current, mapping, "plain")
                current_shingles = _shingles(text)
                duplicate = any(
                    current_shingles and other and
                    len(current_shingles & other) / max(1, len(current_shingles | other)) >= near_threshold
                    for other in shingles[-300:]
                )
                if duplicate:
                    stats["near_duplicates_dropped"] += 1
                    continue
                shingles.append(current_shingles)
            if current != original:
                stats["records_modified"] += 1
            kept += 1
            yield current

    write_records(generator(), destination, fmt)
    report = {
        "input_records": total,
        "output_records": kept,
        "dropped": total - kept,
        "operation_stats": dict(stats),
        "output": str(destination),
        "operations": operations,
        "finished_at": now_iso(),
    }
    write_json_atomic(Path(f"{destination}.audit.json"), report)
    return report


def split_dataset(
    path: Path,
    mapping: dict[str, Any],
    output_dir: Path,
    ratios: dict[str, float] | None = None,
    counts: dict[str, int] | None = None,
    seed: int = 42,
    fmt: str = "jsonl",
    grouped_by: str | None = None,
) -> dict[str, Any]:
    """Deterministic seeded split into train/validation/test files."""
    ratios = ratios or {"train": 0.9, "validation": 0.05, "test": 0.05}
    names = list(ratios.keys())
    ensure_dir(output_dir)
    rows = list(iter_records(path))
    if grouped_by:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in rows:
            key = str(record.get(grouped_by, "__none__"))
            groups.setdefault(key, []).append(record)
        rng = random.Random(seed)
        keys = sorted(groups.keys())
        rng.shuffle(keys)
        total_groups = len(keys)
        bounds: dict[str, tuple[int, int]] = {}
        cursor = 0
        for index, name in enumerate(names):
            share = counts.get(name) if counts else round(total_groups * ratios[name])
            share = max(0, min(total_groups - cursor, int(share or 0) if index < len(names) - 1 else total_groups - cursor))
            bounds[name] = (cursor, cursor + share)
            cursor += share
        buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for index, key in enumerate(keys):
            for name in names:
                start, end = bounds[name]
                if start <= index < end:
                    buckets[name].extend(groups[key])
                    break
    else:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        total = len(indices)
        buckets = {}
        cursor = 0
        for index, name in enumerate(names):
            remaining = total - cursor
            share = counts.get(name) if counts else round(total * ratios[name])
            share = int(share or 0)
            share = min(remaining, share) if index < len(names) - 1 else remaining
            chosen = indices[cursor:cursor + share]
            buckets[name] = [rows[i] for i in chosen]
            cursor += share
    written: dict[str, str] = {}
    for name in names:
        target = output_dir / f"{safe_slug(name)}.{fmt if fmt != 'parquet' else 'parquet'}"
        write_records(buckets[name], target, fmt)
        written[name] = str(target)
    report = {
        "seed": seed,
        "ratios": ratios,
        "counts": {name: len(buckets[name]) for name in names},
        "files": written,
        "grouped_by": grouped_by,
        "created_at": now_iso(),
    }
    write_json_atomic(output_dir / "split.json", report)
    return report


def export_dataset(path: Path, mapping: dict[str, Any], destination: Path, fmt: str) -> dict[str, Any]:
    write_records((normalise_record(record, mapping) or record for record in iter_records(path)), destination, fmt)
    size, _count = dir_size(destination)
    return {"output": str(destination), "format": fmt, "size_bytes": size, "size_human": human_bytes(size)}


def dataset_card(path: Path, report: dict[str, Any] | None = None, notes: str = "") -> str:
    report = report or inspect_dataset(path)
    lines = [
        f"# Dataset card — {report.get('name', Path(path).name)}",
        "",
        f"- Path: `{report.get('path')}`",
        f"- Format: {report.get('kind')}",
        f"- Records: {report.get('record_count')}",
        f"- Size: {report.get('size_human')}",
        f"- Fields: {', '.join(report.get('field_names') or []) or 'n/a'}",
        f"- Duplicate records (exact): {report.get('duplicates')}",
        f"- Empty records: {report.get('empty_records')}",
        f"- Average characters: {report.get('length', {}).get('average')}",
        f"- Estimated tokens: {(report.get('token_estimate') or {}).get('total')} (estimated, characters / 4)",
        "",
        "## Preprocessing",
        "",
        "Prepared with ZeqouXTraining. Cleaning operations and audits are stored next to the "
        "dataset files as `<file>.audit.json`.",
        "",
        "## Known limitations",
        "",
        notes.strip() or "Not documented yet.",
        "",
        f"Generated at {now_iso()}.",
    ]
    return "\n".join(lines)


def stored_metadata(path: Path) -> dict[str, Any]:
    return read_json(Path(f"{path}.meta.json"), {}) or {}
