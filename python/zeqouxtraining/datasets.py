"""Dataset loading, field auto-detection, normalisation and validation.

Everything here is real file I/O. Only the Hugging Face *datasets* path needs an
optional dependency; JSON/JSONL/CSV/TXT/folders work on bare Python, which means
a dataset can be validated on any machine.

:func:`analyze` returns two parallel things:

* ``samples``  — only the usable samples, ready for training
* ``statuses`` — one entry per *raw* record, so statistics never lose alignment
  when a row in the middle is dropped
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------- #
# Supported sources
# --------------------------------------------------------------------------- #
#: Extension -> canonical format id. This table *is* the contract: the desktop
#: app asks for it over the protocol (``dataset-formats``) and only offers what
#: is listed here, so the interface can never advertise a format the backend
#: would refuse to open.
FORMAT_BY_EXTENSION: dict[str, str] = {
    # Structured text
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".jsonlines": "jsonl",
    # Tables
    ".csv": "csv",
    ".psv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    # Plain text
    ".txt": "txt",
    ".text": "txt",
    ".md": "txt",
    ".markdown": "txt",
    # Columnar
    ".parquet": "parquet",
    ".pq": "parquet",
    ".arrow": "arrow",
    ".feather": "arrow",
    ".orc": "orc",
    # Databases
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".db": "sqlite",
    # Spreadsheets
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    # YAML
    ".yaml": "yaml",
    ".yml": "yaml",
}

#: Compressed variants are read transparently: ``shard-01.jsonl.gz`` is a JSONL
#: dataset that happens to be gzipped.
COMPRESSION_SUFFIXES: dict[str, str] = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".lzma": "xz",
}

SUPPORTED_EXTENSIONS = set(FORMAT_BY_EXTENSION)
TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown"}

#: Single-file formats ``export-dataset`` can write.
EXPORT_FORMATS = ("jsonl", "json", "csv", "tsv", "txt", "parquet")

#: format id -> (human label, reading dependency, hint when the dependency is
#: missing). ``None`` means the reader ships with Python itself.
FORMAT_INFO: dict[str, tuple[str, str | None, str]] = {
    "json": ("JSON array, or an object with a list inside", None, ""),
    "jsonl": ("JSON Lines / NDJSON — one record per line", None, ""),
    "csv": ("CSV — comma/semicolon/pipe separated with a header", None, ""),
    "tsv": ("TSV / TAB — tab separated with a header", None, ""),
    "txt": ("Plain text or Markdown — split into paragraphs", None, ""),
    "yaml": ("YAML list of records", "yaml", "pip install pyyaml"),
    "parquet": ("Parquet columnar table", "pyarrow", "pip install pyarrow"),
    "arrow": ("Arrow IPC / Feather table", "pyarrow", "pip install pyarrow"),
    "orc": ("ORC columnar table", "pyarrow", "pip install pyarrow"),
    "excel": ("Excel workbook (.xlsx/.xlsm/.xls)", "openpyxl", "pip install openpyxl"),
    "sqlite": ("SQLite database — the largest table is read", None, ""),
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def supported_formats() -> dict[str, Any]:
    """The formats this install can actually read, for the UI to display."""
    formats = []
    for fmt, (label, requires, hint) in FORMAT_INFO.items():
        formats.append({
            "id": fmt,
            "label": label,
            "requires": requires,
            "available": requires is None or _has_module(requires),
            "hint": "" if requires is None or _has_module(requires) else hint,
            "extensions": sorted(
                extension for extension, mapped in FORMAT_BY_EXTENSION.items() if mapped == fmt
            ),
        })
    return {
        "formats": formats,
        "extensions": sorted(FORMAT_BY_EXTENSION),
        "compression": sorted(COMPRESSION_SUFFIXES),
        "export_formats": list(EXPORT_FORMATS),
        "sources": ["local file", "local folder of shards", "sqlite database", "huggingface hub"],
        "hub_available": _has_module("datasets"),
    }

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

def split_compression(name: str) -> tuple[str, str | None]:
    """``shard.jsonl.gz`` -> ``("shard.jsonl", "gzip")``."""
    lowered = name.lower()
    for suffix, codec in COMPRESSION_SUFFIXES.items():
        if lowered.endswith(suffix):
            return name[: -len(suffix)], codec
    return name, None


def is_supported_file(name: str) -> bool:
    """True for a file name this loader can open, compressed or not."""
    stem, _codec = split_compression(Path(name).name)
    return Path(stem).suffix.lower() in SUPPORTED_EXTENSIONS


def detect_format(path: str) -> str:
    target = Path(path)
    if target.is_dir():
        return "folder"
    stem, _codec = split_compression(target.name)
    suffix = Path(stem).suffix.lower()
    fmt = FORMAT_BY_EXTENSION.get(suffix)
    if fmt is None:
        if _codec:
            # shard.jsonl.gz is readable; shard.gz is not, because the name no
            # longer says what is inside it.
            raise DatasetError(
                f"'{target.name}' is compressed, but the name does not say which format is inside.",
                hint=f"Rename it with the inner extension, e.g. 'shard.jsonl{target.suffix.lower()}'. "
                     "Supported inside archives: "
                     + ", ".join(sorted(set(FORMAT_BY_EXTENSION) & {
                         ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"
                     }))
                     + ".",
                code="unsupported_format",
            )
        raise DatasetError(
            f"Unsupported file type '{target.suffix.lower() or target.name}'.",
            hint="Supported: "
                 + ", ".join(sorted(FORMAT_BY_EXTENSION))
                 + ", a folder of shards, or a Hugging Face dataset id.",
            code="unsupported_format",
        )
    return fmt


def detect_compression(path: str) -> str | None:
    _stem, codec = split_compression(Path(path).name)
    return codec


def _decompress(raw: bytes, codec: str, name: str) -> bytes:
    try:
        if codec == "gzip":
            return gzip.decompress(raw)
        if codec == "bz2":
            return bz2.decompress(raw)
        return lzma.decompress(raw)
    except (OSError, EOFError, lzma.LZMAError) as exc:
        raise DatasetError(
            f"'{name}' could not be decompressed: {exc}.",
            hint="The file looks truncated or is not really compressed. Re-download it intact.",
            code="bad_compression",
        ) from exc


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(
            f"Could not read '{path.name}': {exc.strerror or exc}.",
            hint="Check that the file exists and is not locked by another program.",
            code="unreadable",
        ) from exc

    codec = detect_compression(path.name)
    if codec:
        raw = _decompress(raw, codec, path.name)

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
        "compression": detect_compression(target.name),
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
    elif resolved == "tsv":
        records = _load_csv(target, max_records, delimiter="\t")
        meta["files"] = [target.name]
    elif resolved == "parquet":
        records = _load_parquet(target, max_records)
        meta["files"] = [target.name]
    elif resolved == "arrow":
        records = _load_arrow(target, max_records)
        meta["files"] = [target.name]
    elif resolved == "orc":
        records = _load_orc(target, max_records)
        meta["files"] = [target.name]
    elif resolved == "sqlite":
        records, table = _load_sqlite(target, max_records)
        meta["items"] = table
        meta["files"] = [target.name]
    elif resolved == "excel":
        records = _load_excel(target, max_records, meta)
        meta["files"] = [target.name]
    elif resolved == "yaml":
        records = _load_yaml(target, max_records)
        meta["files"] = [target.name]
    else:
        raise DatasetError(
            f"Unknown dataset format '{resolved}'.",
            hint="Supported types are listed in Settings → Datasets.",
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
            if is_supported_file(name):
                candidates.append(Path(root) / name)

    if not candidates:
        raise DatasetError(
            f"No supported dataset files inside '{target.name}'.",
            hint="Folders may contain " + ", ".join(sorted(FORMAT_BY_EXTENSION)) + " files.",
            code="empty_folder",
        )

    meta["files"] = [str(p.relative_to(target)).replace("\\", "/") for p in candidates]
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


def _load_csv(target: Path, max_records: int, delimiter: str | None = None) -> list[Any]:
    text = _read_text(target)
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
    else:
        # An explicit delimiter keeps a .tsv file that happens to contain commas
        # from being split down the wrong column.
        dialect = csv.excel
        dialect.delimiter = delimiter

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


def _load_arrow(target: Path, max_records: int) -> list[Any]:
    """Arrow IPC and Feather files are the same table format, different framing."""
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.feather as feather  # noqa: PLC0415
        import pyarrow.ipc as ipc  # noqa: PLC0415
    except Exception as exc:
        raise DatasetError(
            "Reading .arrow/.feather datasets requires the 'pyarrow' package.",
            hint="Install it with: pip install pyarrow",
            code="missing_dependency",
        ) from exc

    try:
        try:
            with pa.memory_map(str(target), "r") as source:
                table = ipc.open_file(source).read_all()
        except Exception:
            table = feather.read_table(target)
        rows = table.slice(0, max_records).to_pylist()
    except Exception as exc:
        raise DatasetError(f"Could not read '{target.name}' as Arrow: {exc}", code="arrow_error") from exc
    if not rows:
        raise DatasetError("The Arrow file contains no rows.", code="empty")
    return rows


def _load_orc(target: Path, max_records: int) -> list[Any]:
    try:
        import pyarrow.orc as orc  # noqa: PLC0415
    except Exception as exc:
        raise DatasetError(
            "Reading .orc datasets requires the 'pyarrow' package.",
            hint="Install it with: pip install pyarrow",
            code="missing_dependency",
        ) from exc
    try:
        rows = orc.read_table(target).slice(0, max_records).to_pylist()
    except Exception as exc:
        raise DatasetError(f"Could not read '{target.name}' as ORC: {exc}", code="orc_error") from exc
    if not rows:
        raise DatasetError("The ORC file contains no rows.", code="empty")
    return rows


def _load_sqlite(target: Path, max_records: int) -> tuple[list[Any], str | None]:
    """Read the biggest table of a SQLite database.

    SQLite needs no third-party package, which makes a ``.db`` export a very
    portable way to hand someone a dataset.
    """
    import sqlite3  # noqa: PLC0415 - stdlib, imported here to keep the top clean

    try:
        connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise DatasetError(
            f"Could not open '{target.name}' as a SQLite database: {exc}",
            hint="Only real SQLite .db/.sqlite files can be read.",
            code="sqlite_error",
        ) from exc

    try:
        connection.row_factory = sqlite3.Row
        names = [
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        if not names:
            raise DatasetError(
                f"'{target.name}' holds no tables.",
                hint="Point at the database that contains your records.",
                code="empty",
            )

        def row_count(table: str) -> int:
            try:
                return int(connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0])
            except sqlite3.Error:
                return 0

        table = max(names, key=row_count)
        rows = [dict(row) for row in connection.execute(
            f'SELECT * FROM "{table}" LIMIT {int(max_records)}'
        ).fetchall()]
    except DatasetError:
        raise
    except sqlite3.Error as exc:
        raise DatasetError(f"Could not read '{target.name}': {exc}", code="sqlite_error") from exc
    finally:
        connection.close()

    if not rows:
        raise DatasetError(f"Table '{table}' in '{target.name}' is empty.", code="empty")
    return [_json_safe_row(row) for row in rows], table


def _load_excel(target: Path, max_records: int, meta: dict[str, Any]) -> list[Any]:
    try:
        from openpyxl import load_workbook  # noqa: PLC0415
    except Exception as exc:
        raise DatasetError(
            "Reading .xlsx/.xlsm datasets requires the 'openpyxl' package.",
            hint="Install it with: pip install openpyxl",
            code="missing_dependency",
        ) from exc

    try:
        workbook = load_workbook(target, read_only=True, data_only=True)
    except Exception as exc:
        raise DatasetError(
            f"Could not read '{target.name}' as a spreadsheet: {exc}",
            hint="For legacy .xls files, save the sheet as .xlsx first.",
            code="excel_error",
        ) from exc

    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if not header:
            raise DatasetError(f"'{target.name}' is empty.", code="empty")
        columns = [str(name) if name is not None else f"column_{index}"
                   for index, name in enumerate(header)]
        meta["items"] = sheet.title
        records = []
        for row in rows:
            if row is None or all(value is None for value in row):
                continue
            records.append({
                columns[index] if index < len(columns) else f"column_{index}": value
                for index, value in enumerate(row)
            })
            if len(records) >= max_records:
                break
    finally:
        workbook.close()

    if not records:
        raise DatasetError(f"Sheet '{sheet.title}' in '{target.name}' has no data rows.", code="empty")
    return [_json_safe_row(row) for row in records]


def _load_yaml(target: Path, max_records: int) -> list[Any]:
    try:
        import yaml  # noqa: PLC0415
    except Exception as exc:
        raise DatasetError(
            "Reading .yaml/.yml datasets requires the 'pyyaml' package.",
            hint="Install it with: pip install pyyaml",
            code="missing_dependency",
        ) from exc

    try:
        parsed = yaml.safe_load(_read_text(target))
    except Exception as exc:
        raise DatasetError(f"'{target.name}' is not valid YAML: {exc}", code="invalid_yaml") from exc

    if isinstance(parsed, list):
        return parsed[:max_records]
    if isinstance(parsed, dict):
        picked, _key = _pick_list_from_object(parsed)
        if picked is not None:
            return picked[:max_records]
        return [parsed]
    raise DatasetError(
        f"'{target.name}' contains a {type(parsed).__name__}, which cannot be used as a dataset.",
        hint="A YAML dataset must be a list of records, or a mapping with a list inside.",
        code="invalid_shape",
    )


def _json_safe_row(row: Any) -> Any:
    """bytes/memoryview out of a database become readable strings."""
    if isinstance(row, dict):
        return {
            key: (value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray, memoryview)) else value)
            for key, value in row.items()
        }
    return row


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
    """Load records from either a local path or a Hugging Face dataset id."""
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


def render_pair(instruction: str, extra_input: str, output: str, template_text: str) -> str:
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
    """Convert raw records into samples plus a per-record status list."""
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
# Single-file export
# --------------------------------------------------------------------------- #

def _scalar(value: Any) -> Any:
    """Anything a CSV cell can hold: nested values survive as JSON text."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def flatten_row(row: Any) -> dict[str, Any]:
    """One exportable record: chat roles are rendered to text, the rest is kept."""
    if isinstance(row, str):
        return {"text": row}
    if not isinstance(row, dict):
        return {"text": json.dumps(row, ensure_ascii=False)}

    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, list) and value and isinstance(value[0], dict) \
                and ("role" in value[0] or "from" in value[0]):
            messages = _normalize_messages(value)
            flat[key] = json.dumps(messages, ensure_ascii=False) if messages else _scalar(value)
        else:
            flat[key] = _scalar(value)
    return flat


def _export_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """Stable column order: first-seen across every row."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns, [[row.get(column) for column in columns] for row in rows]


def _write_export(target: Path, fmt: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if fmt == "jsonl":
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"fields": sorted({key for row in rows for key in row})}

    if fmt == "json":
        # A plain array: every reader of JSON datasets understands it.
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return {"fields": sorted({key for row in rows for key in row})}

    if fmt in ("csv", "tsv"):
        columns, table = _export_rows(rows)
        with open(target, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t" if fmt == "tsv" else ",",
                                lineterminator="\n")
            writer.writerow(columns)
            for line in table:
                writer.writerow(["" if value is None else value for value in line])
        return {"fields": columns}

    if fmt == "txt":
        text = "\n\n".join(
            str(row.get("text") or "")
            for row in rows
            if str(row.get("text") or "").strip()
        )
        if not text:
            raise DatasetError(
                "Plain text export needs a single text column.",
                hint="Export as JSONL instead — this dataset is made of several fields.",
                code="not_text",
            )
        target.write_text(text + "\n", encoding="utf-8")
        return {"fields": ["text"]}

    if fmt == "parquet":
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except Exception as exc:
            raise DatasetError(
                "Writing .parquet requires the 'pyarrow' package.",
                hint="Install it with: pip install pyarrow, or export as JSONL.",
                code="missing_dependency",
            ) from exc
        columns, _table = _export_rows(rows)
        table = pa.table({column: [row.get(column) for row in rows] for column in columns})
        pq.write_table(table, target)
        return {"fields": columns}

    raise DatasetError(
        f"'{fmt}' is not an export format.",
        hint="Export formats: " + ", ".join(EXPORT_FORMATS) + ".",
        code="unsupported_format",
    )


def export_dataset(
    path: str | None = None,
    *,
    hf_id: str | None = None,
    split: str = "train",
    fmt: str = "auto",
    output: str = "",
    output_format: str | None = None,
    mapping: dict[str, Any] | None = None,
    raw: bool = False,
    max_records: int = 200_000,
    overwrite: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Write the whole dataset — file, folder of shards or Hub id — as ONE file.

    Normalised content is the default: what the trainer would actually see,
    after field detection and templating. ``raw=True`` keeps the original
    records untouched, for handing the data to another tool.
    """
    if not str(output or "").strip():
        raise DatasetError(
            "Choose where the exported file should be written.",
            hint="Pick a file name ending in " + ", ".join(EXPORT_FORMATS) + ".",
            code="no_output",
        )

    target = Path(str(output)).expanduser()
    if target.is_dir():
        raise DatasetError(
            f"'{target}' is a folder.",
            hint="An export is a single file: give it a name such as dataset.jsonl.",
            code="output_is_dir",
        )

    resolved_format = (output_format or "").strip().lower()
    if not resolved_format:
        stem, _codec = split_compression(target.name)
        resolved_format = FORMAT_BY_EXTENSION.get(Path(stem).suffix.lower(), "")
    if resolved_format not in EXPORT_FORMATS:
        raise DatasetError(
            f"Cannot export to '{target.suffix or target.name}'.",
            hint="Export formats: " + ", ".join(EXPORT_FORMATS) + ".",
            code="unsupported_format",
        )

    if target.exists() and not overwrite:
        raise DatasetError(
            f"'{target.name}' already exists.",
            hint="Choose another name, or allow the export to replace that file.",
            code="exists",
        )

    records, meta = resolve_source(
        path=path,
        hf_id=hf_id,
        split=split,
        fmt=fmt,
        max_records=max_records,
        progress=progress,
    )
    resolved_mapping = mapping or detect_mapping(records)

    kind = resolved_mapping.get("kind", "unknown")
    if raw or kind == "unknown":
        rows = [flatten_row(record) for record in records]
        mode = "raw"
    else:
        samples = normalize(records, resolved_mapping)
        rows = [flatten_row(sample) for sample in samples]
        mode = "normalised"

    if not rows:
        raise DatasetError(
            "Nothing to export: no usable records were produced.",
            hint="Validate the dataset first — the report says which records were dropped and why.",
            code="empty",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    written = _write_export(target, resolved_format, rows)

    # Plain text has no record boundary other than a blank line, so say what that
    # means instead of letting a later read surprise anyone.
    note = ""
    if resolved_format == "txt":
        note = ("Plain text separates records with a blank line, so a sample that contains one "
                "becomes two when the file is read back. JSONL keeps record boundaries exactly.")

    return {
        "output": str(target),
        "name": target.name,
        "format": resolved_format,
        "mode": mode,
        "note": note,
        "records": len(rows),
        "source_records": len(records),
        "bytes": target.stat().st_size,
        "fields": written["fields"],
        "mapping": resolved_mapping,
        "source": meta,
    }


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
                "compression": None,
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
            "compression": meta.get("compression"),
            "items": meta.get("items"),
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
