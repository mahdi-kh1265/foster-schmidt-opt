"""Synthesize / Optimize page."""

from __future__ import annotations

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.gui.view_models.optimize_vm import OptimizeVM
from foster_eom.gui.workers.base_worker import BaseWorker


class SynthesizePage(QWidget):
    """P05 optimization page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._result = None
        self._worker: BaseWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Controls ---
        ctrl = QHBoxLayout()
        self.btn_run = QPushButton("Run Optimization (P05)")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)
        ctrl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress, 1)

        self.lbl_status = QLabel("Idle")
        ctrl.addWidget(self.lbl_status)
        lay.addLayout(ctrl)

        # --- Result summary ---
        summary_grp = QGroupBox("Result")
        summary_lay = QVBoxLayout(summary_grp)
        self.lbl_result_status = QLabel("—")
        self.lbl_result_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        summary_lay.addWidget(self.lbl_result_status)
        lay.addWidget(summary_grp)

        # --- Candidate table ---
        cand_grp = QGroupBox("Candidates")
        cand_lay = QVBoxLayout(cand_grp)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Objective",
                "Feasible",
                "Near-Feasible",
                "Numerical Status",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.currentCellChanged.connect(self._on_selection)
        cand_lay.addWidget(self.table)
        lay.addWidget(cand_grp, 1)

        # --- Selected candidate details ---
        detail_grp = QGroupBox("Selected Candidate Details")
        detail_lay = QVBoxLayout(detail_grp)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(180)
        detail_lay.addWidget(self.detail_text)
        lay.addWidget(detail_grp)

    # ------------------------------------------------------------------
    def set_state(self, state: ProjectState) -> None:
        self._state = state
        self.btn_run.setEnabled(True)

    def set_stale(self) -> None:
        self.lbl_status.setText("⚠ Results stale — inputs changed")
        self.lbl_status.setStyleSheet("color: orange;")

    @property
    def result(self):
        return self._result

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self._state:
            return
        self._result = None
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running…")
        self.lbl_status.setStyleSheet("")
        self.table.setRowCount(0)
        self.detail_text.clear()
        self.lbl_result_status.setText("—")

        state = self._state

        worker = BaseWorker(
            fn=lambda: OptimizeCtrl.run(state),
            project_revision=state.revision,
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self, result: object) -> None:
        self._result = result
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

        from foster_eom.optimize.engine import OptimizationResult

        if isinstance(result, OptimizationResult):
            vm = OptimizeVM.from_result(result)
            self.lbl_result_status.setText(vm.status_label)

            color = {"FEASIBLE": "green", "NEAR-FEASIBLE": "#cc8800", "INFEASIBLE": "red"}
            self.lbl_result_status.setStyleSheet(
                f"font-weight: bold; font-size: 14px; color: {color.get(vm.status_label, 'black')};"
            )
            self.lbl_status.setText(f"Done — {len(vm.candidates)} candidates")
            self.lbl_status.setStyleSheet("")

            self.table.setRowCount(len(vm.candidates))
            for i, c in enumerate(vm.candidates):
                self.table.setItem(i, 0, QTableWidgetItem(str(c.rank)))
                self.table.setItem(i, 1, QTableWidgetItem(f"{c.objective:.6f}"))
                self.table.setItem(i, 2, QTableWidgetItem("✓" if c.feasible else "✗"))
                self.table.setItem(i, 3, QTableWidgetItem("✓" if c.near_feasible else "✗"))
                self.table.setItem(i, 4, QTableWidgetItem(c.numerical_status))

    def _on_error(self, err_type: str, err_msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Error: {err_type}")
        self.lbl_status.setStyleSheet("color: red;")
        self.lbl_result_status.setText("ERROR")
        self.lbl_result_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
        self.detail_text.setPlainText(err_msg)

    def _on_selection(self, row: int, *_args) -> None:
        from foster_eom.optimize.engine import OptimizationResult

        if not isinstance(self._result, OptimizationResult):
            return
        if 0 <= row < len(self._result.candidates):
            c = self._result.candidates[row]
            lines = [
                f"Candidate #{row + 1}",
                f"Topology: {c.topology_id}",
                f"Objective: {c.base_objective_value:.6f} (base) + {c.soft_penalty_total:.6f} (soft)",
                f"Feasible: {c.feasible}  |  V_max: {c.v_max:.4f}",
                f"Numerical: {c.numerical_status}",
                f"Seed source: {c.seed_source}",
                "",
                "Objective terms:",
            ]
            for k, v in c.objective_terms.items():
                lines.append(f"  {k}: {v:.6f}")
            lines.append("")
            lines.append("Constraint margins:")
            for k, v in c.constraint_margins.items():
                lines.append(f"  {k}: {v:.4f}")
            self.detail_text.setPlainText("\n".join(lines))
