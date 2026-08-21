"""P12-GUI2 Single-Run Lifecycle Regression Tests."""

from unittest.mock import MagicMock, patch

from PySide6 import QtCore

from foster_eom.gui.pages.synthesize_page import SynthesizePage
from foster_eom.gui.state import ProjectState
from foster_eom.optimize.engine import OptimizationResult

from pytestqt.qtbot import QtBot

def test_single_run_lifecycle_emits_once(qtbot: QtBot) -> None:
    """
    Proves:
    - one Run click
    - exactly one OptimizeWorker launched
    - exactly one finished lifecycle / result render
    - Run disabled while active, re-enabled after.
    """
    state = ProjectState()
    state.name = "Lifecycle Test"

    page = SynthesizePage()
    page.set_state(state)
    qtbot.addWidget(page)

    with patch.object(page, "_on_finished", wraps=page._on_finished) as mock_finished:
        with patch("foster_eom.gui.pages.synthesize_page.QThreadPool.start") as mock_start:
            # Click Run
            assert page.btn_run.isEnabled()
            qtbot.mouseClick(page.btn_run, QtCore.Qt.MouseButton.LeftButton)

            # Exactly one worker queued to thread pool
            mock_start.assert_called_once()

            # Should be disabled now
            assert not page.btn_run.isEnabled()

            # Get the worker instance passed to start
            worker = mock_start.call_args[0][0]

            # Simulate the thread doing work then emitting finished
            worker.signals.finished.emit(OptimizationResult(
                candidates=(),
                best_feasible=None,
                near_feasible_best=None,
                preflight=MagicMock(),
                seed_diagnostics=MagicMock(),
                de_diagnostics=(),
                run_manifest=MagicMock(requested_global_budget=10, random_seed=42)
            ))

            # Emitted finished exactly once
            mock_finished.assert_called_once()

            # Re-enabled after finish
            assert page.btn_run.isEnabled()
