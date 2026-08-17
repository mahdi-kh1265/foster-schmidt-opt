"""Base Qt worker for executing backend tasks."""

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal


class WorkerSignals(QObject):
    """Signals for background workers."""

    progress = Signal(int, str)  # -1 for indeterminate progress
    finished = Signal(object)
    error = Signal(str, str)


class BaseWorker(QRunnable):
    """Executes a function on the thread pool with revision tracking."""

    def __init__(self, fn, project_revision: str, library_sha: str | None = None):
        super().__init__()
        self.fn = fn
        self.project_revision = project_revision
        self.library_sha = library_sha
        self.signals = WorkerSignals()
        self.cancelled = False

    def cancel(self):
        """Mark as cancelled. Checked before dispatch."""
        self.cancelled = True

    def run(self):
        if self.cancelled:
            return

        self.signals.progress.emit(-1, "Running…")
        try:
            # We don't pass a cancel_event to the backend because frozen P01-P11
            # APIs don't cooperatively support it.
            result = self.fn()

            # If cancelled while backend was running, discard the result
            if self.cancelled:
                return

            self.signals.finished.emit(result)
        except Exception as e:
            if self.cancelled:
                return
            err_type = type(e).__name__
            err_msg = f"{e}\n\n{traceback.format_exc()}"
            self.signals.error.emit(err_type, err_msg)
