"""Realization / P09 page."""

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
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.controllers.realization_ctrl import RealizationCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.gui.workers.base_worker import BaseWorker


class RealizationPage(QWidget):
    """P09 catalog realization page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._opt_result = None
        self._result = None
        self._worker: BaseWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Controls ---
        ctrl = QHBoxLayout()
        self.btn_run = QPushButton("Run Realization (P09)")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)
        ctrl.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress, 1)
        self.lbl_status = QLabel("Idle — requires optimization + library")
        ctrl.addWidget(self.lbl_status)
        lay.addLayout(ctrl)

        # --- Summary ---
        summary_grp = QGroupBox("Realization Status")
        summary_lay = QVBoxLayout(summary_grp)
        self.lbl_result_status = QLabel("—")
        self.lbl_result_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        summary_lay.addWidget(self.lbl_result_status)
        self.lbl_degradation = QLabel("")
        summary_lay.addWidget(self.lbl_degradation)
        lay.addWidget(summary_grp)

        # --- Slot mapping table ---
        slot_grp = QGroupBox("Element → Catalog Part Mapping")
        slot_lay = QVBoxLayout(slot_grp)
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                "Element",
                "Vendor",
                "Part Number",
                "Nominal Value",
                "Tolerance",
                "Model Tier",
                "log(ratio)",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        slot_lay.addWidget(self.table)
        lay.addWidget(slot_grp, 1)

    # ------------------------------------------------------------------
    def set_prereqs(self, state: ProjectState, opt_result: object) -> None:
        self._state = state
        self._opt_result = opt_result
        has_opt = opt_result is not None
        has_lib = bool(state.library_path)
        self.btn_run.setEnabled(has_opt and has_lib)
        if not has_opt:
            self.lbl_status.setText("Idle — requires optimization result")
        elif not has_lib:
            self.lbl_status.setText("Idle — requires library")
        else:
            self.lbl_status.setText("Ready")

    def set_stale(self) -> None:
        self.lbl_status.setText("⚠ Results stale")
        self.lbl_status.setStyleSheet("color: orange;")

    @property
    def result(self):
        return self._result

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self._state or not self._opt_result:
            return
        self._result = None
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running…")
        self.lbl_status.setStyleSheet("")
        self.table.setRowCount(0)
        self.lbl_result_status.setText("—")
        self.lbl_degradation.setText("")

        state = self._state
        opt = self._opt_result

        worker = BaseWorker(
            fn=lambda: RealizationCtrl.run(state, opt),
            project_revision=state.revision,
            library_sha=state.library_sha,
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self, result: object) -> None:
        self._result = result
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

        from foster_eom.realization.result import RealizationResult

        if not isinstance(result, RealizationResult):
            return

        status_map = {
            "feasible": ("FEASIBLE", "green"),
            "degraded": ("DEGRADED", "#cc8800"),
            "infeasible": ("INFEASIBLE (exhaustive search)", "red"),
            "no_feasible_found": ("NO FEASIBLE FOUND (beam search limit)", "red"),
            "no_candidates": ("NO CANDIDATES", "red"),
        }
        label, color = status_map.get(result.status, (result.status.upper(), "black"))
        self.lbl_result_status.setText(label)
        self.lbl_result_status.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")
        self.lbl_status.setText("Done")
        self.lbl_status.setStyleSheet("color: green;")

        if result.degradation is not None and result.continuous_baseline.objective_value != 0:
            pct = (result.degradation / abs(result.continuous_baseline.objective_value)) * 100.0
            self.lbl_degradation.setText(
                f"Objective degradation: {result.degradation:.6f} "
                f"({pct:.1f}% of continuous baseline)"
            )
        else:
            self.lbl_degradation.setText("")

        # Populate slot table
        if result.best is not None:
            entries = list(result.best.slot_entries.items())
            self.table.setRowCount(len(entries))
            for i, (eid, ne) in enumerate(entries):
                self.table.setItem(i, 0, QTableWidgetItem(eid))
                self.table.setItem(i, 1, QTableWidgetItem(ne.vendor))
                self.table.setItem(i, 2, QTableWidgetItem(ne.part_number))
                self.table.setItem(i, 3, QTableWidgetItem(f"{ne.value_nom:.3e}"))
                tol = f"±{ne.value_tol_frac * 100:.1f}%" if ne.value_tol_frac else "—"
                self.table.setItem(i, 4, QTableWidgetItem(tol))
                self.table.setItem(i, 5, QTableWidgetItem(ne.model_tier.value))
                self.table.setItem(i, 6, QTableWidgetItem(f"{ne.log_ratio:.4f}"))
        else:
            self.table.setRowCount(0)

    def _on_error(self, err_type: str, err_msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Error: {err_type}")
        self.lbl_status.setStyleSheet("color: red;")
        self.lbl_result_status.setText("ERROR")
        self.lbl_result_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
