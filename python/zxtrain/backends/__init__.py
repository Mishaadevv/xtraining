"""Training / inference backends.

Each backend declares what it can do on the current machine. The registry hides
nothing: a backend that needs an uninstalled runtime is reported as unavailable
together with the reason and the exact package list required.
"""

from __future__ import annotations

from .base import Backend, BackendInfo, TrainSpec

__all__ = ["Backend", "BackendInfo", "TrainSpec"]
