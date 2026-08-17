"""Robustness / P10 page."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.controllers.robustness_ctrl import RobustnessCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.gui.workers.base_worker import BaseWorker

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _MPL_OK = True
except ImportError:
    _MPL_OK = False


class RobustnessPage(QWidget):
    """P10 robustness analysis page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._real_result = None
        self._worker: BaseWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Controls ---
        ctrl = QHBoxLayout()
        self.btn_run = QPushButton("Run Robustness (P10)")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)
        ctrl.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress, 1)
        self.lbl_status = QLabel("Idle — requires realization")
        ctrl.addWidget(self.lbl_status)
        lay.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Yield Summary ---
        yield_grp = QGroupBox("Yield Summary")
        yield_lay = QVBoxLayout(yield_grp)

        self.lbl_evaluable = QLabel("Evaluable Yield: —")
        self.lbl_evaluable.setStyleSheet("font-weight: bold; font-size: 14px;")
        yield_lay.addWidget(self.lbl_evaluable)

        self.lbl_bounds = QLabel("Overall Yield Bounds: —")
        yield_lay.addWidget(self.lbl_bounds)

        self.lbl_ci = QLabel("")
        yield_lay.addWidget(self.lbl_ci)

        # Failure counts
        self.lbl_counts = QLabel("")
        yield_lay.addWidget(self.lbl_counts)

        # P06 diagnostic (not yield)
        self.lbl_p06 = QLabel("")
        self.lbl_p06.setStyleSheet("color: #555;")
        yield_lay.addWidget(self.lbl_p06)

        splitter.addWidget(yield_grp)

        # --- Sensitivity bar chart ---
        sens_grp = QGroupBox("OAT Sensitivity (Dominant Slots)")
        sens_lay = QVBoxLayout(sens_grp)

        if _MPL_OK:
            self.fig_sens = Figure(figsize=(5, 3), dpi=100)
            self.canvas_sens = FigureCanvasQTAgg(self.fig_sens)
            sens_lay.addWidget(self.canvas_sens)
        else:
            # Fallback: table
            self.sens_table = QTableWidget()
            self.sens_table.setColumnCount(2)
            self.sens_table.setHorizontalHeaderLabels(["Slot", "Sensitivity"])
            self.sens_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            sens_lay.addWidget(self.sens_table)

        splitter.addWidget(sens_grp)
        lay.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    def set_prereqs(self, state: ProjectState, real_result: object) -> None:
        self._state = state
        self._real_result = real_result
        enabled = real_result is not None and state.library_path is not None
        self.btn_run.setEnabled(enabled)
        if enabled:
            self.lbl_status.setText("Ready")
        else:
            self.lbl_status.setText("Idle — requires realization + library")

    def set_stale(self) -> None:
        self.lbl_status.setText("⚠ Results stale")
        self.lbl_status.setStyleSheet("color: orange;")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self._state or not self._real_result:
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running…")
        self.lbl_status.setStyleSheet("")

        state = self._state
        real = self._real_result

        worker = BaseWorker(
            fn=lambda: RobustnessCtrl.run(state, real),
            project_revision=state.revision,
            library_sha=state.library_sha,
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self, result: object) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Done")
        self.lbl_status.setStyleSheet("color: green;")

        from foster_eom.gui.view_models.robustness_vm import RobustnessVM

        vm = RobustnessVM.from_result(result)
        s = vm.summary

        # Yield
        self.lbl_evaluable.setText(f"Evaluable Yield: {s.evaluable_yield_pct:.1f}%")
        color = (
            "green"
            if s.evaluable_yield_pct > 90
            else ("#cc8800" if s.evaluable_yield_pct > 50 else "red")
        )
        self.lbl_evaluable.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")

        self.lbl_bounds.setText(
            f"Overall Yield Bounds: [{s.yield_lower_bound_pct:.1f}%, "
            f"{s.yield_upper_bound_pct:.1f}%]"
        )

        # Wilson CI
        if s.ci_displayed and s.ci_lo_pct is not None and s.ci_hi_pct is not None:
            self.lbl_ci.setText(
                f"Wilson {s.ci_level_pct:.0f}% CI: [{s.ci_lo_pct:.1f}%, {s.ci_hi_pct:.1f}%]"
            )
        else:
            self.lbl_ci.setText("Wilson CI: not available (non-iid sampling)")

        # Counts
        self.lbl_counts.setText(
            f"Samples: {s.n_samples}  |  "
            f"Pass: {s.n_pass}  |  "
            f"Physical Fail: {s.n_physical_fail}  |  "
            f"Model Unresolved: {s.n_model_unresolved}  |  "
            f"Numerical Unresolved: {s.n_numerical_unresolved}"
        )

        # P06 diagnostic — explicitly labeled as diagnostic, not yield
        if s.p06_diagnostic_label:
            self.lbl_p06.setText(
                f"P06 Worst-K Diagnostic (not a yield estimator): {s.p06_diagnostic_label}"
            )
        else:
            self.lbl_p06.setText("")

        # Sensitivity chart
        if _MPL_OK and vm.sensitivity:
            self.fig_sens.clear()
            ax = self.fig_sens.add_subplot(111)
            slots = [s.slot for s in vm.sensitivity[:10]]  # top 10
            impacts = [s.impact for s in vm.sensitivity[:10]]
            y_pos = range(len(slots))
            ax.barh(y_pos, impacts, align="center", color="#3377bb")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(slots, fontsize=8)
            ax.set_xlabel("Sensitivity (ΔJ)")
            ax.set_title("OAT Sensitivity — Dominant Slots")
            ax.invert_yaxis()
            self.fig_sens.tight_layout()
            self.canvas_sens.draw()
        elif not _MPL_OK and hasattr(self, "sens_table") and vm.sensitivity:
            self.sens_table.setRowCount(len(vm.sensitivity))
            for i, s in enumerate(vm.sensitivity):
                self.sens_table.setItem(i, 0, QTableWidgetItem(s.slot))
                self.sens_table.setItem(i, 1, QTableWidgetItem(f"{s.impact:.6f}"))

    def _on_error(self, err_type: str, err_msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Error: {err_type}")
        self.lbl_status.setStyleSheet("color: red;")
