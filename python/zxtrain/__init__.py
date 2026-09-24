"""ZeqouXTraining engine.

A local-first engine for model inspection, dataset preparation, training,
continuation training, evaluation, conversion and local serving.

The engine is deliberately dependency-light: every operation that can be done
with the Python standard library is done with the standard library, and optional
ML runtimes (PyTorch, Transformers, PEFT, ...) are detected at runtime. Nothing
is ever reported as available when it is not installed.
"""

from __future__ import annotations

ENGINE_NAME = "zxtrain"
ENGINE_VERSION = "2.0.0"
PROTOCOL_VERSION = 1

__all__ = ["ENGINE_NAME", "ENGINE_VERSION", "PROTOCOL_VERSION"]
