"""Read safetensors metadata without importing torch.

The format is a little-endian u64 header length, a JSON header, then the tensor
payload. Parsing the header alone gives exact parameter counts, dtypes and
shapes for any checkpoint — including ones too large to load.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

DTYPE_BITS = {
    "F64": 64, "I64": 64, "U64": 64,
    "F32": 32, "I32": 32, "U32": 32,
    "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8, "I8": 8, "U8": 8, "BOOL": 8,
}


def read_header(path: Path, max_header: int = 200 * 1024 * 1024) -> dict[str, Any]:
    """Return {'metadata': {...}, 'tensors': {name: {...}}} for one file."""
    path = Path(path)
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            return {"metadata": {}, "tensors": {}, "error": "file is shorter than a safetensors header"}
        (length,) = struct.unpack("<Q", length_bytes)
        if length <= 0 or length > max_header:
            return {"metadata": {}, "tensors": {}, "error": f"implausible header length {length}"}
        raw = handle.read(length)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"metadata": {}, "tensors": {}, "error": f"header is not valid JSON: {exc}"}
    metadata = header.pop("__metadata__", {}) or {}
    return {"metadata": metadata, "tensors": header, "error": None}


def tensor_elements(entry: dict[str, Any]) -> int:
    count = 1
    for dim in entry.get("shape", []) or []:
        try:
            count *= int(dim)
        except (TypeError, ValueError):
            return 0
    return count


def summarise(files: list[Path]) -> dict[str, Any]:
    """Aggregate params / dtypes / total bytes across one or more weight files."""
    total_params = 0
    total_bytes = 0
    dtypes: dict[str, int] = {}
    tensors = 0
    errors: list[str] = []
    metadata: dict[str, str] = {}
    for path in files:
        header = read_header(path)
        if header.get("error"):
            errors.append(f"{path.name}: {header['error']}")
            continue
        for key, value in (header.get("metadata") or {}).items():
            metadata.setdefault(str(key), str(value))
        for _name, entry in (header.get("tensors") or {}).items():
            if not isinstance(entry, dict):
                continue
            elements = tensor_elements(entry)
            dtype = str(entry.get("dtype", "?"))
            bits = DTYPE_BITS.get(dtype, 0)
            total_params += elements
            total_bytes += elements * bits // 8
            dtypes[dtype] = dtypes.get(dtype, 0) + elements
            tensors += 1
    return {
        "parameter_count": total_params,
        "weight_bytes": total_bytes,
        "tensor_count": tensors,
        "dtypes": dtypes,
        "metadata": metadata,
        "parsed_files": [str(p) for p in files],
        "errors": errors,
    }
