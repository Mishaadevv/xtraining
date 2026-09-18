"""Training backends.

Importing this package must stay cheap and dependency-free: backends import
their heavy libraries lazily inside ``run``.
"""

from .base import RunContext, RunResult, TrainingBackend
from .registry import available_backends, backend_capabilities, get_backend

__all__ = [
    "RunContext",
    "RunResult",
    "TrainingBackend",
    "available_backends",
    "backend_capabilities",
    "get_backend",
]
