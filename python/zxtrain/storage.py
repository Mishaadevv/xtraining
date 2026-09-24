"""Workspace storage: what is using disk, what is orphaned, what can be cleaned.

Detection is read-only. Deleting requires an explicit `confirm=True` call from the
UI, and protected artefacts (marks written by the user) are never removed.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .errors import ZxError
from .util import dir_size, ensure_dir, human_bytes, now_iso, read_json

WORKSPACE_DIRS = ("models", "datasets", "jobs", "exports", "runtime", "cache", "logs", "projects", "servers")


def report(workspace: str | Path, extra_paths: list[str] | None = None) -> dict[str, Any]:
    workspace = Path(workspace)
    entries: list[dict[str, Any]] = []
    total = 0
    for name in WORKSPACE_DIRS:
        path = workspace / name
        if not path.exists():
            entries.append({"name": name, "path": str(path), "exists": False, "bytes": 0, "files": 0,
                            "human": "0 B"})
            continue
        size, files = dir_size(path)
        total += size
        entries.append({
            "name": name, "path": str(path), "exists": True, "bytes": size, "files": files,
            "human": human_bytes(size),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
        })
    for extra in (extra_paths or []):
        path = Path(extra)
        if path.exists():
            size, files = dir_size(path)
            entries.append({"name": path.name, "path": str(path), "exists": True, "bytes": size,
                            "files": files, "human": human_bytes(size), "external": True})
    disk_free = None
    try:
        usage = shutil.disk_usage(str(workspace))
        disk_free = {"total": usage.total, "free": usage.free, "used_percent": round(usage.used / usage.total * 100, 1)}
    except OSError:
        pass
    protected: list[dict[str, Any]] = []
    for name in ("models", "datasets", "exports"):
        folder = workspace / name
        if not folder.is_dir():
            continue
        for child in sorted(folder.iterdir()):
            if not child.is_dir():
                continue
            marker = read_json(child / "zxtrain-model.json", {}) or {}
            if marker.get("protected"):
                protected.append({
                    "path": str(child),
                    "name": child.name,
                    "kind": name,
                    "since": marker.get("updated_at"),
                })
    categories = {entry["name"]: entry["bytes"] for entry in entries}
    return {
        "workspace": str(workspace),
        "entries": sorted(entries, key=lambda entry: entry["bytes"], reverse=True),
        "categories": categories,
        "total_bytes": total,
        "total_human": human_bytes(total),
        "disk": disk_free,
        "volume": str(Path(workspace).anchor) or None,
        "volume_bytes": (disk_free or {}).get("total"),
        "free_bytes": (disk_free or {}).get("free"),
        "protected": protected,
        "reported_at": now_iso(),
        "generated_at": now_iso(),
    }


def tree(path: str | Path, depth: int = 2, max_entries: int = 500) -> dict[str, Any]:
    root = Path(path).expanduser()
    if not root.exists():
        raise ZxError(code="path_missing", message=f"Path does not exist: {root}",
                      hint="Check the folder name or pick another location.")
    entries: list[dict[str, Any]] = []

    def walk(current: Path, level: int) -> None:
        if len(entries) >= max_entries:
            return
        try:
            children = sorted(current.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        except OSError:
            return
        for child in children:
            if len(entries) >= max_entries:
                return
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "relative": child.relative_to(root).as_posix(),
                "is_dir": child.is_dir(),
                "size": stat.st_size,
                "size_human": human_bytes(stat.st_size),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "depth": level,
            })
            if child.is_dir() and level < depth:
                walk(child, level + 1)

    walk(root, 0)
    return {"root": str(root), "entries": entries, "truncated": len(entries) >= max_entries}


def orphans(workspace: str | Path) -> dict[str, Any]:
    """Find artefacts nothing points at any more. Nothing is removed here."""
    workspace = Path(workspace)
    jobs_root = workspace / "jobs"
    referenced_models: set[str] = set()
    referenced_checkpoints: set[str] = set()
    job_dirs: list[Path] = []

    if jobs_root.exists():
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            job_dirs.append(job_dir)
            spec = read_json(job_dir / "spec.json", {}) or {}
            status = read_json(job_dir / "status.json", {}) or {}
            for key in ("base_model", "parent_checkpoint", "resume_from"):
                value = spec.get(key)
                if value:
                    referenced_models.add(str(Path(str(value))))
            result = status.get("result") or {}
            if result.get("model_dir"):
                referenced_models.add(str(Path(str(result["model_dir"]))))
            for checkpoint in status.get("checkpoints") or []:
                if checkpoint.get("path"):
                    referenced_checkpoints.add(str(Path(str(checkpoint["path"]))))

    findings: list[dict[str, Any]] = []
    for path in (workspace / "models").glob("*") if (workspace / "models").exists() else []:
        if not path.is_dir():
            continue
        marker = read_json(path / "zxtrain-model.json", {}) or {}
        if marker.get("protected"):
            continue
        if str(path.resolve()) not in {str(Path(item).resolve()) for item in referenced_models}:
            size, files = dir_size(path)
            findings.append({
                "kind": "model",
                "path": str(path),
                "name": path.name,
                "bytes": size,
                "human": human_bytes(size),
                "files": files,
                "reason": "No training run or lineage record references this model folder. "
                          "It may be an imported base model.",
                "safe": False,
            })

    temp_files = []
    for pattern in ("*.tmp", "*.part", ".zxtrain-*"):
        temp_files.extend(workspace.rglob(pattern))
        if len(temp_files) > 50:
            break
    for path in temp_files:
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                continue
            findings.append({
                "kind": "temporary",
                "path": str(path),
                "name": path.name,
                "bytes": size,
                "human": human_bytes(size),
                "reason": "Temporary file left by an interrupted operation.",
                "safe": True,
            })

    dead_jobs = []
    for job_dir in job_dirs:
        status = read_json(job_dir / "status.json", {}) or {}
        if status.get("state") in ("interrupted", "failed") and not (status.get("resumable_from")):
            size, files = dir_size(job_dir)
            dead_jobs.append({
                "kind": "job",
                "path": str(job_dir),
                "name": job_dir.name,
                "bytes": size,
                "human": human_bytes(size),
                "files": files,
                "reason": f"Job ended as {status.get('state')} and has no resumable checkpoint.",
                "safe": True,
            })
    findings.extend(dead_jobs)

    return {
        "workspace": str(workspace),
        "findings": findings,
        "total_bytes": sum(item["bytes"] for item in findings if item.get("safe")),
        "total_human": human_bytes(sum(item["bytes"] for item in findings if item.get("safe"))),
        "checked_at": now_iso(),
        "note": "Nothing is deleted automatically. Entries marked safe can be removed from this page.",
    }


def clean(paths: list[str], confirm: bool, allow_jobs: bool = True) -> dict[str, Any]:
    """Remove only what the caller explicitly listed, and only with confirm=True."""
    if not confirm:
        raise ZxError(
            code="confirmation_required",
            message="Refusing to delete without an explicit confirmation.",
            hint="Tick the confirmation box in the storage panel; deletions are never silent.",
        )
    removed: list[str] = []
    skipped: list[str] = []
    freed = 0
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            skipped.append(f"{raw} (already gone)")
            continue
        resolved = path.resolve()
        # Refuse anything that is not clearly an app artefact.
        if not any(part in ("models", "datasets", "jobs", "exports", "cache", "logs") for part in resolved.parts):
            skipped.append(f"{raw} (outside the workspace artefact folders)")
            continue
        marker = read_json(resolved / "zxtrain-model.json", {}) or {}
        if marker.get("protected"):
            skipped.append(f"{raw} (protected)")
            continue
        if resolved.is_dir() and not allow_jobs and "jobs" in resolved.parts:
            skipped.append(f"{raw} (job folders are locked)")
            continue
        size, _files = dir_size(resolved)
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            removed.append(str(resolved))
            freed += size
        except OSError as exc:
            skipped.append(f"{raw} ({exc})")
    return {"removed": removed, "skipped": skipped, "freed_bytes": freed, "freed_human": human_bytes(freed),
            "finished_at": now_iso()}


def set_protected(path: str | Path, protected: bool) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise ZxError(code="path_missing", message=f"Cannot protect {target}: it does not exist.")
    marker_path = target / "zxtrain-model.json" if target.is_dir() else Path(f"{target}.zxtrain.json")
    marker = read_json(marker_path, {}) or {}
    marker["protected"] = bool(protected)
    marker["updated_at"] = now_iso()
    ensure_dir(marker_path.parent)
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return {"path": str(target), "protected": bool(protected)}
