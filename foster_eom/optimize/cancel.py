"""Cooperative cancellation support for optimization runs.

Provides a lightweight exception and helper for checking a ``threading.Event``
cancel token.  The backend never depends on Qt.
"""

from __future__ import annotations

import threading


class CancelledException(Exception):
    """Raised when cooperative cancellation is detected."""


def check_cancel(event: threading.Event | None) -> None:
    """Raise ``CancelledException`` if *event* is set.

    Safe to call with ``None`` (no-op).
    """
    if event is not None and event.is_set():
        raise CancelledException("Optimization cancelled by user")
