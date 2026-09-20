"""Checkpoint handling.

Checkpoints are ordinary Hugging Face Trainer checkpoints on disk, so a run can
always be resumed by any tool — not only by this app. Everything here reads
real files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

CHECKPOINT_PREFIX = "checkpoint-"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _last_loss(state: dict[str, Any] | None) -> float | None:
    if not state:
        return None
    for entry in reversed(state.get("log_history", [])):
        if "loss" in entry:
            return entry["loss"]
    return None


def _tree_size(root: Path) -> int:
    total = 0
    for file_path in root.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def list_checkpoints(run_dir: str | Path) -> list[dict[str, Any]]:
    """Enumerate checkpoints newest-first, with the info needed to resume."""
    root = Path(run_dir)
    if not root.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(CHECKPOINT_PREFIX):
            continue
        try:
            step = int(child.name[len(CHECKPOINT_PREFIX):])
        except ValueError:
            continue

        state = _read_json(child / "trainer_state.json")
        entries.append({
            "name": child.name,
            "path": str(child.resolve()),
            "step": step,
            "size_bytes": _tree_size(child),
            "created": child.stat().st_mtime,
            "loss": _last_loss(state),
            "epoch": state.get("epoch") if state else None,
            "has_optimizer": any(
                (child / name).is_file() for name in ("optimizer.pt", "optimizer.bin", "optimizer")
            ),
            "has_scheduler": any(
                (child / name).is_file() for name in ("scheduler.pt", "scheduler.bin", "scheduler")
            ),
        })

    entries.sort(key=lambda item: item["step"], reverse=True)
    return entries


def latest_checkpoint(run_dir: str | Path) -> dict[str, Any] | None:
    checkpoints = list_checkpoints(run_dir)
    return checkpoints[0] if checkpoints else None


def prune(run_dir: str | Path, keep: int = 3) -> list[str]:
    """Delete the oldest checkpoints beyond ``keep``. Returns removed paths."""
    if keep <= 0:
        return []
    removed: list[str] = []
    for entry in list_checkpoints(run_dir)[keep:]:
        try:
            shutil.rmtree(entry["path"])
            removed.append(entry["path"])
        except OSError:
            continue
    return removed


def size_of(run_dir: str | Path) -> int:
    root = Path(run_dir)
    if not root.is_dir():
        return 0
    return _tree_size(root)
