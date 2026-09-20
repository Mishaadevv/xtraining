"""Pluggable training backends.

A backend subclasses :class:`TrainingBackend`, declares the packages it
requires, and registers itself in ``registry.py``. The runner, the protocol and
the UI stay unchanged.
"""
