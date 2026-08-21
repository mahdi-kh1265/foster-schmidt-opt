"""Optimization-specific Qt worker with cooperative cancellation and progress.

This is an opt-in worker that does NOT modify ``BaseWorker``.  It creates its
own ``threading.Event`` for cancellation and bridges ``ProgressUpdate`` to the
GUI thread via ``Signal(object)``.
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal

from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.optimize.progress import ProgressUpdate


class OptimizeWorkerSignals(QObject):
    """Signals for the optimization worker."""

    progress = Signal(object)   # ProgressUpdate dataclass
    finished = Signal(object)   # OptimizationResult
    error = Signal(str, str)    # (error_type, traceback)


class OptimizeWorker(QRunnable):
    """Runs ``OptimizeCtrl.run()`` with cooperative cancellation and progress."""

    def __init__(self, state: ProjectState) -> None:
        super().__init__()
        self.state = state
        self.signals = OptimizeWorkerSignals()
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        """Set the cooperative cancellation flag."""
        self.cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self) -> None:
        if self.cancel_event.is_set():
            return

        def _progress_bridge(update: ProgressUpdate) -> None:
            """Thread-safe bridge: emit ProgressUpdate to the Qt GUI thread."""
            self.signals.progress.emit(update)

        try:
            result = OptimizeCtrl.run(
                state=self.state,
                cancel_event=self.cancel_event,
                progress_callback=_progress_bridge,
            )

            if self.cancel_event.is_set():
                # Do not present cancelled result as successful.
                self.signals.progress.emit(
                    ProgressUpdate(phase="CANCELLED", overall_percent=0, phase_percent=0)
                )
                return

            self.signals.finished.emit(result)
        except Exception as e:
            if self.cancel_event.is_set():
                return
            err_type = type(e).__name__
            err_msg = f"{e}\n\n{traceback.format_exc()}"
            self.signals.error.emit(err_type, err_msg)
