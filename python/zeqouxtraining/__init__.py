"""ZeqouXTraining Python backend.

The backend is driven by the Electron main process as a subprocess. It speaks a
single, explicit wire protocol:

* **stdout** carries newline-delimited JSON events only (see ``events.py``).
* **stderr** carries human-readable log lines, including anything third-party
  libraries print.

That separation matters: training libraries are chatty, and mixing their output
into stdout would corrupt the protocol. Importing this package must never import
torch — the diagnostics commands have to run on a bare Python install.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
