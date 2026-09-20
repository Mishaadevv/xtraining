"""ZeqouXTraining Python backend.

The backend runs as a subprocess of the Electron main process and speaks one
explicit wire protocol:

* **stdout** — newline-delimited JSON events only (see ``events.py``);
* **stderr** — human-readable output, including anything third-party prints.

Importing this package never imports torch: diagnostics must work on a bare
Python install so the app can explain what is missing instead of crashing.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
