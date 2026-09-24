"""Precision conversion and quantization planning.

Two things live here and they are deliberately kept apart:

* :func:`plan` — an honest compatibility/estimate report. It says what the
  machine can actually do right now (bitsandbytes present? llama.cpp converter
  on PATH?) and what the result would cost in bytes. Estimates are labelled as
  such; nothing is invented.
* :func:`convert` — a genuine dtype conversion for safetensors checkpoints,
  implemented without torch so it also works on a CPU-only install. It rewrites
  the header and the payload; shapes, names and file layout stay intact.

int4/int8 *quantization* (weights plus scales, different loader) is a different
operation and is delegated to bitsandbytes — when that is missing the plan says
so instead of pretending.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
from array import array
from pathlib import Path
from typing import Any, Callable

from . import models
from .errors import ZxError
from .safetensors import DTYPE_BITS, read_header
from .util import ensure_dir, human_bytes, now_iso

FLOAT_DTYPES = ("F32", "F16", "BF16")
#: dtypes this module can physically write without torch.
CONVERTIBLE = {
    "F32": ("F32", "F16", "BF16"),
    "F16": ("F16", "F32", "BF16"),
    "BF16": ("BF16", "F32", "F16"),
}

GGUF_CONVERTER_NAMES = (
    "convert_hf_to_gguf.py",
    "convert-hf-to-gguf.py",
    "convert.py",
)


def _numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency, checked at call time
    except Exception:  # noqa: BLE001 - any import failure means "no numpy"
        return None
    return numpy


def _find_gguf_converter(workspace: str | Path | None = None) -> str | None:
    """Look for a llama.cpp converter script in the workspace or on PATH."""
    candidates: list[Path] = []
    if workspace:
        root = Path(workspace)
        for name in GGUF_CONVERTER_NAMES:
            candidates.append(root / "tools" / name)
            candidates.append(root / "tools" / "llama.cpp" / name)
    for name in GGUF_CONVERTER_NAMES:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def tool_availability(workspace: str | Path | None = None) -> dict[str, Any]:
    """Which conversion/quantization tools this interpreter can reach."""
    import importlib.util

    def spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    converter = _find_gguf_converter(workspace)
    return {
        "bitsandbytes": {
            "available": spec("bitsandbytes"),
            "used_for": "int8/int4 weight quantization (loads with transformers)",
            "install": "pip install bitsandbytes",
        },
        "torch": {
            "available": spec("torch"),
            "used_for": "loading PyTorch checkpoints and writing .bin/.pt weights",
            "install": "see the Environment page",
        },
        "numpy": {
            "available": spec("numpy"),
            "used_for": "fast dtype conversion (the pure-Python path is used otherwise)",
            "install": "pip install numpy",
        },
        "gguf_converter": {
            "available": converter is not None,
            "path": converter,
            "used_for": "Hugging Face → GGUF conversion (llama.cpp)",
            "install": "place convert_hf_to_gguf.py in <workspace>/tools/ or add it to PATH",
        },
        "safetensors": {
            "available": spec("safetensors"),
            "used_for": "third-party safetensors readers; the built-in header reader is used regardless",
            "install": "pip install safetensors",
        },
        "native_precision": {
            "available": True,
            "path": "built-in",
            "used_for": "fp32/fp16/bf16 safetensors conversion without torch",
            "install": "already available",
        },
    }


def _dtype_bytes(dtypes: dict[str, int]) -> dict[str, int]:
    return {dtype: count * DTYPE_BITS.get(dtype, 0) // 8 for dtype, count in dtypes.items()}


def _header_overhead(root: Path) -> int:
    """Bytes spent on safetensors headers, measured on the actual files."""
    total = 0
    try:
        files = models.list_weight_files(root).get("safetensors") or []
    except OSError:
        return 0
    for item in files:
        try:
            with Path(item).open("rb") as handle:
                length_bytes = handle.read(8)
            if len(length_bytes) == 8:
                total += 8 + struct.unpack("<Q", length_bytes)[0]
        except (OSError, struct.error):
            continue
    return total


def plan(
    model_path: str | Path,
    target_dtype: str = "F16",
    target_format: str = "safetensors",
    workspace: str | Path | None = None,
    keep_metadata: bool = True,
) -> dict[str, Any]:
    """Describe what converting this model would produce, or explain why not."""
    target_dtype = (target_dtype or "F16").upper()
    target_format = (target_format or "safetensors").lower()
    report = models.inspect(str(model_path))
    weights = report.get("weights") or {}
    dtypes: dict[str, int] = dict(weights.get("dtypes") or {})
    source_bytes = int(weights.get("weight_bytes") or 0)
    parameters = int(weights.get("parameter_count") or 0)
    tools = tool_availability(workspace)

    checks: list[dict[str, Any]] = []

    def check(name: str, status: str, message: str, hint: str = "") -> None:
        checks.append({"name": name, "status": status, "message": message, "hint": hint})

    convertible = [dtype for dtype in dtypes if dtype in CONVERTIBLE]
    unknown = sorted(dtype for dtype in dtypes if dtype not in CONVERTIBLE)
    if not dtypes:
        check("dtypes", "Unsupported", "No readable weight dtypes were found in this model.",
              "Import a safetensors or PyTorch checkpoint to convert its precision.")
    for dtype in convertible:
        if dtype != target_dtype:
            if target_dtype in CONVERTIBLE[dtype]:
                check(f"{dtype}→{target_dtype}", "Supported", f"{dtype} tensors can be rewritten as {target_dtype}.")
            else:
                check(f"{dtype}→{target_dtype}", "Unsupported", f"{dtype} cannot be written as {target_dtype}.")
    for dtype in unknown:
        check(dtype, "Unsupported", f"{dtype} tensors are left untouched by the built-in converter.",
              "Only fp32/fp16/bf16 are converted losslessly by this path.")

    # Estimated output size: every float tensor moves to the target dtype, other
    # tensors (integers, booleans) keep their width.
    if parameters and source_bytes:
        byte_map = {dtype: DTYPE_BITS.get(dtype, 0) // 8 for dtype in dtypes}
        estimated = 0
        for dtype, count in dtypes.items():
            width = byte_map.get(target_dtype, byte_map.get(dtype, 0)) if dtype in CONVERTIBLE else byte_map.get(dtype, 0)
            estimated += count * width
        estimated += _header_overhead(Path(model_path))
    else:
        estimated = source_bytes

    format_support: dict[str, Any]
    if target_format == "safetensors":
        format_support = {"status": "Supported", "tool": "built-in writer", "available": True}
    elif target_format == "gguf":
        available = tools["gguf_converter"]["available"]
        format_support = {
            "status": "Supported" if available else "Unsupported",
            "tool": "llama.cpp convert_hf_to_gguf.py",
            "available": available,
            "path": tools["gguf_converter"]["path"],
            "hint": "" if available else tools["gguf_converter"]["install"],
        }
    elif target_format in ("bin", "pytorch", "pt"):
        available = tools["torch"]["available"]
        format_support = {
            "status": "Supported" if available else "Unsupported",
            "tool": "torch.save",
            "available": available,
            "hint": "" if available else "Install the PyTorch runtime from the Environment page.",
        }
    else:
        format_support = {"status": "Unsupported", "tool": target_format, "available": False,
                          "hint": "Choose safetensors, gguf or pytorch."}

    # int8/int4 need a quantization backend with scales, not a dtype rewrite.
    if target_dtype in ("INT8", "INT4", "I8", "Q8", "Q4"):
        supported = tools["bitsandbytes"]["available"]
        check("quantization backend", "Supported" if supported else "Unsupported",
              "bitsandbytes is installed and can produce int8/int4 weights."
              if supported else "int4/int8 weights need bitsandbytes.",
              "" if supported else tools["bitsandbytes"]["install"])

    statuses = {entry["status"] for entry in checks}
    overall = "Unsupported" if statuses == {"Unsupported"} else ("Supported with limitations" if "Unsupported" in statuses else "Supported")
    if target_format != "safetensors" and not format_support["available"]:
        overall = "Unsupported"

    return {
        "source": str(model_path),
        "target_dtype": target_dtype,
        "target_format": target_format,
        "source_dtype_bytes": _dtype_bytes(dtypes),
        "source_bytes": source_bytes,
        "source_human": human_bytes(source_bytes),
        "estimated_bytes": int(estimated),
        "estimated_human": human_bytes(int(estimated)),
        "estimated": True,
        "parameter_count": parameters,
        "ratio": round(estimated / source_bytes, 4) if source_bytes else None,
        "checks": checks,
        "status": overall,
        "format_support": format_support,
        "tools": tools,
        "metadata_preserved": bool(keep_metadata),
        "notes": [
            "Output size is an estimate: it assumes every float tensor moves to the target dtype.",
            "The source model is never modified; conversion writes to a new folder.",
        ],
        "generated_at": now_iso(),
    }


# --------------------------------------------------------------------------- #
# Real dtype conversion for safetensors
# --------------------------------------------------------------------------- #

def _to_f32_bytes(raw: bytes, source: str, elements: int) -> bytes:
    """Widen fp16/bf16 payloads to little-endian fp32."""
    if source == "F32":
        return raw
    if source == "F16":
        # struct's 'e' is IEEE binary16 and always available, unlike array('e').
        return array("f", struct.unpack(f"<{elements}e", raw)).tobytes()
    if source == "BF16":
        # bfloat16 is the top 16 bits of the fp32 bit pattern.
        out = bytearray(elements * 4)
        out[2::4] = raw[0::2]
        out[3::4] = raw[1::2]
        return bytes(out)
    raise ZxError(code="unsupported_dtype", message=f"Cannot read dtype {source}.")


def _from_f32_bytes(raw: bytes, target: str, elements: int) -> bytes:
    """Narrow a little-endian fp32 payload to fp16/bf16."""
    if target == "F32":
        return raw
    if target == "F16":
        values = struct.unpack(f"<{elements}f", raw)
        return struct.pack(f"<{elements}e", *values)
    if target == "BF16":
        # Truncating the low mantissa bits is exactly what bf16 means; the engine
        # says so in the conversion report rather than pretending it is lossless.
        out = bytearray(elements * 2)
        out[0::2] = raw[2::4]
        out[1::2] = raw[3::4]
        return bytes(out)
    raise ZxError(code="unsupported_dtype", message=f"Cannot write dtype {target}.")


def _numpy_rewrite(raw: bytes, source: str, target: str):
    numpy = _numpy()
    if numpy is None:
        return None
    if source == "BF16":
        values = (numpy.frombuffer(raw, dtype="<u2").astype("uint32") << 16).view("<f4")
    elif source == "F16":
        values = numpy.frombuffer(raw, dtype="<f2")
    else:
        values = numpy.frombuffer(raw, dtype="<f4")
    if target == "BF16":
        words = numpy.asarray(values, dtype="<f4").view("<u4")
        return (words >> 16).astype("<u2").tobytes()
    if target == "F16":
        return values.astype("<f2").tobytes()
    return values.astype("<f4").tobytes()


def rewrite_tensor(raw: bytes, source: str, target: str, elements: int) -> bytes:
    """Convert one tensor payload, or return it unchanged when not convertible."""
    if source == target or source not in CONVERTIBLE or target not in CONVERTIBLE.get(source, ()):
        return raw
    converted = _numpy_rewrite(raw, source, target)
    if converted is not None:
        return converted
    if source == "F32":
        return _from_f32_bytes(raw, target, elements)
    widened = _to_f32_bytes(raw, source, elements)
    return _from_f32_bytes(widened, target, elements)


def convert_safetensors(path: Path, destination: Path, target_dtype: str,
                        progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Rewrite one safetensors file with a different float dtype."""
    header = read_header(path)
    if header.get("error"):
        raise ZxError(code="unreadable_weights", message=f"{path.name}: {header['error']}",
                      hint="The file is not a valid safetensors checkpoint.")
    tensors: dict[str, dict[str, Any]] = header["tensors"]
    metadata: dict[str, Any] = header.get("metadata") or {}

    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        handle.seek(8 + header_length)
        payload = handle.read()

    new_header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    converted = 0
    untouched: list[str] = []
    dtype_counts: dict[str, int] = {}
    for name in sorted(tensors):
        entry = tensors[name]
        start, end = int(entry["data_offsets"][0]), int(entry["data_offsets"][1])
        raw = payload[start:end]
        source_dtype = str(entry.get("dtype"))
        elements = 1
        for dim in entry.get("shape", []) or []:
            elements *= int(dim)
        if source_dtype in CONVERTIBLE and target_dtype in CONVERTIBLE.get(source_dtype, ()):
            if source_dtype != target_dtype:
                data = rewrite_tensor(raw, source_dtype, target_dtype, elements)
                converted += 1
                new_dtype = target_dtype
            else:
                data = raw
                new_dtype = source_dtype
        else:
            data = raw
            new_dtype = source_dtype
            untouched.append(name)
        dtype_counts[new_dtype] = dtype_counts.get(new_dtype, 0) + elements
        new_header[name] = {
            "dtype": new_dtype,
            "shape": entry.get("shape", []),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        chunks.append(data)

    ensure_dir(destination.parent)
    body = b"".join(chunks)
    header_json = json.dumps({**({"__metadata__": metadata} if metadata else {}), **new_header},
                             separators=(",", ":"), sort_keys=False).encode("utf-8")
    # safetensors pads the header so the payload starts 8-byte aligned.
    padding = (-len(header_json)) % 8
    header_json += b" " * padding
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_json)))
        handle.write(header_json)
        handle.write(body)
    os.replace(temporary, destination)
    if progress:
        progress({"type": "log", "level": "info", "message":
                  f"{path.name}: rewrote {converted} tensors to {target_dtype}, {len(untouched)} left untouched."})
    return {
        "file": str(destination),
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "bytes": destination.stat().st_size,
        "converted_tensors": converted,
        "untouched_tensors": len(untouched),
        "untouched_names": untouched[:40],
        "dtype_counts": dtype_counts,
    }


def convert(model_path: str | Path, destination: str | Path, target_dtype: str = "F16",
            target_format: str = "safetensors", workspace: str | Path | None = None,
            progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run a real conversion. Refuses anything the plan marked unsupported."""
    model_path = Path(model_path)
    destination = Path(destination)
    if not model_path.exists():
        raise ZxError(code="model_missing", message=f"Model path does not exist: {model_path}")
    try:
        if destination.resolve() == model_path.resolve():
            raise ZxError(code="unsafe_destination", message="The destination is the source model.",
                          hint="Choose a new folder: the original files are never overwritten.")
    except OSError:
        pass
    if model_path in destination.parents:
        raise ZxError(code="unsafe_destination", message="The destination is inside the source model.",
                      hint="Pick a folder outside the model directory.")

    report = plan(model_path, target_dtype, target_format, workspace)
    format_support = report["format_support"]
    if not format_support.get("available"):
        raise ZxError(
            code="tool_unavailable",
            message=f"{target_format} conversion needs {format_support.get('tool')}.",
            hint=format_support.get("hint") or "Install the required tool, then try again.",
            context={"status": format_support.get("status")},
        )

    emitted: list[dict[str, Any]] = []
    def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        if progress:
            progress(event)

    ensure_dir(destination)
    started = time.time()
    files: list[dict[str, Any]] = []
    copied: list[str] = []

    if target_format == "safetensors":
        weight_files = [Path(item) for item in models.list_weight_files(model_path).get("safetensors") or []]
        if not weight_files:
            raise ZxError(code="no_weights", message="No safetensors files were found in this model.",
                          hint="Only safetensors checkpoints can be converted by the built-in converter.")
        for path in weight_files:
            files.append(convert_safetensors(path, destination / path.name, target_dtype, emit))
    else:
        converter = format_support.get("path") or _find_gguf_converter(workspace)
        if not converter:
            raise ZxError(code="tool_unavailable", message="The GGUF converter could not be located.",
                          hint=format_support.get("hint") or tool_availability(workspace)["gguf_converter"]["install"])
        import subprocess  # noqa: PLC0415 - only needed for the external tool path

        command = [sys_executable(), str(converter), str(model_path),
                   "--outfile", str(destination / f"{model_path.name}.gguf")]
        emit({"type": "log", "level": "info", "message": f"Running {os.path.basename(str(converter))}…"})
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)  # noqa: S603 - fixed argv
        tail = (completed.stdout or "")[-2000:]
        if completed.returncode != 0:
            raise ZxError(code="conversion_failed",
                          message=f"{os.path.basename(str(converter))} exited with code {completed.returncode}.",
                          detail=(completed.stderr or tail)[-2000:],
                          hint="The plan page shows how to install or point at the converter.")
        files.append({"file": str(destination / f"{model_path.name}.gguf"), "tool": str(converter),
                      "log_tail": tail})

    # Keep the small config/tokenizer files so the converted folder still loads.
    for name in models.CONFIG_FILES:
        source = model_path / name
        if source.is_file() and target_format == "safetensors":
            target = destination / name
            if name == "config.json":
                config = models.read_json(source, {}) or {}
                config["torch_dtype"] = target_dtype.lower()
                (target).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
                copied.append(name)
                continue
            shutil.copy2(source, target)
            copied.append(name)

    _write_readme(destination, report, files, copied)
    originals = sum(item["source_bytes"] for item in files if "source_bytes" in item)
    result_size = sum(item.get("bytes", 0) for item in files)
    return {
        "status": "completed",
        "source": str(model_path),
        "destination": str(destination),
        "target_dtype": target_dtype,
        "target_format": target_format,
        "files": files,
        "copied": copied,
        "elapsed_seconds": round(time.time() - started, 2),
        "source_bytes": originals,
        "result_bytes": result_size,
        "plan": report,
        "log": [event.get("message") for event in emitted if event.get("message")],
    }


def sys_executable() -> str:
    import sys

    return sys.executable


def _write_readme(destination: Path, report: dict[str, Any], files: list[dict[str, Any]],
                  copied: list[str]) -> None:
    lines = [
        f"# Converted model ({report['target_format']}, {report['target_dtype']})",
        "",
        f"- Source: `{report['source']}`",
        f"- Converted at: {now_iso()}",
        f"- Estimated output size: {report['estimated_human']} (estimate)",
        f"- Actual output size: {human_bytes(sum(item.get('bytes', 0) for item in files))}",
        "",
        "## Files",
        "",
    ]
    for item in files:
        lines.append(f"- `{Path(item['file']).name}` — {human_bytes(item.get('bytes', 0))}")
    for name in copied:
        lines.append(f"- `{name}` (copied from the source model)")
    lines += [
        "",
        "## Notes",
        "",
        "- The original model was not modified.",
        "- Weights are quantized/dequantized less precisely than the source: quality can change.",
        "- Size figures for the target are estimates; the actual sizes above are measured.",
        "",
    ]
    (destination / "CONVERSION.md").write_text("\n".join(lines), encoding="utf-8")
