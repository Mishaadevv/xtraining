"""External tooling and optional-capability reporting.

Sections 41, 79 and 80 of the product spec ask for tool calling, a local API
and plugins. The honest answer on an arbitrary machine is "it depends", so this
module reports what the *selected model and backend* actually declare instead of
turning features on by default.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .precision import tool_availability
from .util import now_iso, read_json

TOOL_MARKERS = ("tools", "tool_call", "tool_calls", "function", "functions")


def tool_calling(model_path: str | Path | None = None) -> dict[str, Any]:
    """Does this model's own template/config advertise tool calling?"""
    if not model_path:
        return {
            "available": False,
            "reason": "No model is selected.",
            "evidence": [],
            "note": "The engine never invents tool calls; a model must declare them in its own template.",
        }
    root = Path(model_path)
    evidence: list[str] = []
    template = ""
    for name in ("chat_template.jinja", "tokenizer_config.json", "config.json", "generation_config.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
        except OSError:
            continue
        lowered = text.lower()
        if name == "tokenizer_config.json":
            config = read_json(path, {}) or {}
            value = config.get("chat_template")
            if isinstance(value, str):
                template = value
                lowered = value.lower()
        for marker in TOOL_MARKERS:
            if marker in lowered:
                evidence.append(f"{name}: contains “{marker}”")
                break
    available = bool(evidence)
    return {
        "available": available,
        "reason": "The model declares tool/function support in its template or config."
        if available else "This model does not declare tool calling in its template or config.",
        "evidence": evidence[:6],
        "template_preview": template[:400] or None,
        "note": "Tool definitions are passed through to the backend unchanged; nothing is simulated.",
    }


def external_tools(workspace: str | Path | None = None) -> dict[str, Any]:
    """Binaries and services other than Python packages."""
    def which(*names: str) -> str | None:
        for name in names:
            found = shutil.which(name)
            if found:
                return found
        return None

    found = {
        "git": which("git"),
        "git-lfs": which("git-lfs"),
        "ollama": which("ollama"),
        "llama-server": which("llama-server", "llama-cli", "main"),
        "docker": which("docker"),
    }
    return {
        "binaries": {name: {"available": bool(path), "path": path} for name, path in found.items()},
        "relevance": {
            "git-lfs": "needed to download large checkpoints from the Hub",
            "ollama": "alternative local runtime; the app serves models with its own engine instead",
            "llama-server": "alternative GGUF runtime for the Deploy page",
            "docker": "isolated runtime option; not required",
        },
    }


def plugins(workspace: str | Path | None = None) -> dict[str, Any]:
    """Plugin folders the app will load, if the user put any there."""
    if not workspace:
        return {"roots": [], "found": [], "note": "No workspace configured."}
    root = Path(workspace) / "plugins"
    found: list[dict[str, Any]] = []
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            manifest = entry / "plugin.json"
            payload = read_json(manifest, None) if manifest.is_file() else None
            found.append({
                "name": entry.name,
                "path": str(entry),
                "manifest": payload,
                "kind": (payload or {}).get("kind"),
                "compatibility": (payload or {}).get("compatibility"),
                "error": None if payload else "plugin.json is missing or unreadable",
            })
    return {
        "roots": [str(root)],
        "found": found,
        "note": "Plugins are loaded from <workspace>/plugins/<name>/plugin.json; each declares its own compatibility.",
    }


def report(workspace: str | Path | None = None, model_path: str | Path | None = None) -> dict[str, Any]:
    published: dict[str, Any] = {}
    if workspace:
        published_path = Path(workspace) / "api.json"
        if published_path.is_file():
            published = read_json(published_path, {}) or {}
    return {
        "json_mode": json_mode(model_path),
        "checked_at": now_iso(),
        "packages": tool_availability(workspace),
        "external": external_tools(workspace),
        "tool_calling": tool_calling(model_path),
        "plugins": plugins(workspace),
        "local_api": {
            "schema": "OpenAI-compatible (/health, /v1/models, /v1/chat/completions, /v1/completions)",
            "start_hint": "Start it from the Deploy page; the engine binds to 127.0.0.1 by default.",
            "published": published or None,
        },
    }


def json_mode(model_path: str | Path | None = None) -> dict[str, Any]:
    """Structured output support, evidence-based."""
    if not model_path:
        return {"available": False, "reason": "No model selected."}
    root = Path(model_path)
    generation = read_json(root / "generation_config.json", {}) or {}
    config = read_json(root / "config.json", {}) or {}
    hints = [key for key in ("response_format", "json_schema", "guided_decoding") if key in generation or key in config]
    return {
        "available": bool(hints),
        "reason": "The model configuration mentions structured output settings."
        if hints else "Nothing in this model's configuration declares structured output support; "
                      "the engine will not simulate it.",
        "evidence": hints,
        "note": "Even where supported, JSON output depends on the prompt template and sampling settings.",
    }
