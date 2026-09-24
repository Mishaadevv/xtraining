"""Small shared helpers: atomic IO, hashing, formatting, safe paths."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from .errors import ZxError

TEXT_SUFFIXES = {
    ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".psv", ".txt", ".md", ".markdown",
    ".yaml", ".yml", ".html", ".htm", ".parquet", ".arrow", ".feather", ".orc",
    ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".gz", ".bz2", ".xz",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def human_bytes(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def write_json_atomic(path: Path, payload: Any, retries: int = 6) -> None:
    """Write JSON through a temp file + replace so a crash cannot truncate it.

    On Windows ``os.replace`` can fail with ``PermissionError`` when another
    process (an indexer, an antivirus scan, the UI reading the same file) has the
    destination open for a moment. Status files are written many times per
    second during a run, so the write is retried with a short backoff and, if the
    rename still cannot happen, the payload is written in place. Losing
    atomicity is far better than losing the run's status.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:  # includes Windows PermissionError (WinError 5)
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
    try:
        os.unlink(tmp)
    except OSError:
        pass
    del last_error  # the in-place write above succeeded; the rename failure was transient


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_slug(name: str, fallback: str = "item") -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in str(name).strip())
    cleaned = cleaned.strip("-._")
    return cleaned[:80] or fallback


def unique_dir(parent: Path, name: str) -> Path:
    """Return a path under parent that does not exist yet, adding a -2/-3 suffix."""
    base = safe_slug(name)
    candidate = Path(parent) / base
    counter = 2
    while candidate.exists():
        candidate = Path(parent) / f"{base}-{counter}"
        counter += 1
    return candidate


def dir_size(path: Path) -> tuple[int, int]:
    """(bytes, file count) for a file or directory tree."""
    path = Path(path)
    if path.is_file():
        try:
            return path.stat().st_size, 1
        except OSError:
            return 0, 0
    total = 0
    files = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
                files += 1
            except OSError:
                continue
    return total, files


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, limit: int = 400) -> str:
    """A cheap, order-independent fingerprint of a directory's name/size set."""
    root = Path(root)
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                entries.append(f"{path.relative_to(root).as_posix()}:{path.stat().st_size}")
            except OSError:
                continue
        if len(entries) >= limit:
            break
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def guarded_path(path: str | Path, allow_root: bool = False) -> Path:
    """Resolve a user supplied path and refuse obviously destructive targets."""
    resolved = Path(path).expanduser().resolve()
    if not allow_root and resolved == Path(resolved.anchor):
        raise ZxError(
            code="unsafe_path",
            message=f"Refusing to operate on a filesystem root: {resolved}",
            hint="Choose a workspace folder instead of a drive root.",
        )
    return resolved


def iter_text_files(root: Path) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix.lower() in TEXT_SUFFIXES or not path.suffix):
            yield path


def json_safe(value: Any) -> Any:
    """Coerce numpy / torch scalars and other odd types into plain JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    to_python = getattr(value, "tolist", None)
    if callable(to_python):
        try:
            return json_safe(to_python())
        except Exception:  # pragma: no cover - defensive
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)
