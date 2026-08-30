"""Synthesize / Optimize page."""

from __future__ import annotations

import time

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.state import ProjectState
from foster_eom.gui.view_models.optimize_vm import OptimizeVM
from foster_eom.gui.workers.optimize_worker import OptimizeWorker
from foster_eom.optimize.progress import ProgressUpdate


class SynthesizePage(QWidget):
    """P05 optimization page with presets, cancel, and progress."""

    optimization_finished = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._result: object | None = None
        self._worker: OptimizeWorker | None = None
        self._run_start_time: float = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Preset + Controls ---
        ctrl_grp = QGroupBox("Optimization Controls")
        ctrl_lay = QVBoxLayout(ctrl_grp)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["FAST", "BALANCED", "THOROUGH", "CUSTOM"])
        self.preset_combo.setCurrentText("BALANCED")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)
        ctrl_lay.addLayout(preset_row)

        # Custom settings (hidden unless CUSTOM selected)
        self.custom_grp = QGroupBox("Custom Settings")
        custom_lay = QFormLayout(self.custom_grp)
        self.spin_max_evals = QSpinBox()
        self.spin_max_evals.setRange(100, 500_000)
        self.spin_max_evals.setValue(2500)
        custom_lay.addRow("Max Global Evaluations:", self.spin_max_evals)
        self.spin_polish_k = QSpinBox()
        self.spin_polish_k.setRange(1, 20)
        self.spin_polish_k.setValue(2)
        custom_lay.addRow("Polish Top-K:", self.spin_polish_k)
        self.spin_local_iter = QSpinBox()
        self.spin_local_iter.setRange(1, 10_000)
        self.spin_local_iter.setValue(100)
        custom_lay.addRow("Local Max Iterations:", self.spin_local_iter)
        self.custom_grp.setVisible(False)
        ctrl_lay.addWidget(self.custom_grp)

        # Run / Cancel buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Optimization")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_cancel)
        ctrl_lay.addLayout(btn_row)
        lay.addWidget(ctrl_grp)

        # --- Progress panel ---
        progress_grp = QGroupBox("Progress")
        progress_lay = QVBoxLayout(progress_grp)

        self.lbl_phase = QLabel("Phase: Idle")
        self.lbl_phase.setStyleSheet("font-weight: bold;")
        progress_lay.addWidget(self.lbl_phase)

        self.progress_overall = QProgressBar()
        self.progress_overall.setRange(0, 100)
        self.progress_overall.setValue(0)
        self.progress_overall.setFormat("Overall (budget-based estimate): %p%")
        progress_lay.addWidget(self.progress_overall)

        self.progress_phase = QProgressBar()
        self.progress_phase.setRange(0, 100)
        self.progress_phase.setValue(0)
        self.progress_phase.setFormat("Phase: %p%")
        progress_lay.addWidget(self.progress_phase)

        # Detail labels
        detail_lay = QFormLayout()
        self.lbl_de_evals = QLabel("—")
        detail_lay.addRow("DE Evaluations:", self.lbl_de_evals)
        self.lbl_domain = QLabel("—")
        detail_lay.addRow("Domain:", self.lbl_domain)
        self.lbl_polish = QLabel("—")
        detail_lay.addRow("Polish Candidate:", self.lbl_polish)
        self.lbl_local_iter = QLabel("—")
        detail_lay.addRow("Local Iteration:", self.lbl_local_iter)
        self.lbl_elapsed = QLabel("—")
        detail_lay.addRow("Elapsed:", self.lbl_elapsed)
        self.lbl_derivative = QLabel("—")
        detail_lay.addRow("Derivative Mode:", self.lbl_derivative)
        progress_lay.addLayout(detail_lay)

        self.progress_grp = progress_grp
        self.progress_grp.setVisible(False)
        lay.addWidget(progress_grp)

        # --- Result summary ---
        summary_grp = QGroupBox("Result")
        summary_lay = QVBoxLayout(summary_grp)
        self.lbl_result_status = QLabel("—")
        self.lbl_result_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        summary_lay.addWidget(self.lbl_result_status)
        self.lbl_status = QLabel("Idle")
        summary_lay.addWidget(self.lbl_status)
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
        detail_lay2 = QVBoxLayout(detail_grp)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(180)
        detail_lay2.addWidget(self.detail_text)
        lay.addWidget(detail_grp)

    # ------------------------------------------------------------------
    def _on_preset_changed(self, text: str) -> None:
        self.custom_grp.setVisible(text == "CUSTOM")

    def set_state(self, state: ProjectState) -> None:
        self._state = state
        self.btn_run.setEnabled(True)

    def set_stale(self) -> None:
        self.lbl_status.setText("⚠ Results stale — inputs changed")
        self.lbl_status.setStyleSheet("color: orange;")

    @property
    def result(self) -> object:
        return self._result

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self._state:
            return
        self._result = None
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_grp.setVisible(True)
        self.progress_overall.setValue(0)
        self.progress_phase.setValue(0)
        self.lbl_phase.setText("Phase: Starting…")
        self.lbl_status.setText("Running…")
        self.lbl_status.setStyleSheet("")
        self.table.setRowCount(0)
        self.detail_text.clear()
        self.lbl_result_status.setText("—")

        # Write custom values to state if CUSTOM
        preset = self.preset_combo.currentText()
        self._state.optimization_preset.preset = preset
        if preset == "CUSTOM":
            self._state.optimization_preset.custom_max_global_evaluations = self.spin_max_evals.value()
            self._state.optimization_preset.custom_polish_top_k = self.spin_polish_k.value()
            self._state.optimization_preset.custom_local_max_iterations = self.spin_local_iter.value()

        self._run_start_time = time.time()

        worker = OptimizeWorker(self._state)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(self._on_progress)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_phase.setText("Phase: Cancelling…")

    def _on_progress(self, update: ProgressUpdate) -> None:
        """Update progress UI from a ProgressUpdate dataclass."""
        self.lbl_phase.setText(f"Phase: {update.phase}")
        self.progress_overall.setValue(max(0, min(100, update.overall_percent)))
        self.progress_phase.setValue(max(0, min(100, update.phase_percent)))

        if update.de_budget > 0:
            self.lbl_de_evals.setText(f"{update.de_evals:,} / {update.de_budget:,}")
        if update.domain_count > 0:
            self.lbl_domain.setText(f"{update.domain_index + 1} / {update.domain_count}")
        if update.polish_top_k > 0:
            self.lbl_polish.setText(f"{update.polish_candidate_index + 1} / {update.polish_top_k}")
        if update.polish_max_iterations > 0:
            self.lbl_local_iter.setText(
                f"{update.polish_iteration} / {update.polish_max_iterations}"
            )
        elapsed = update.elapsed_s
        if elapsed > 0:
            m, s = divmod(int(elapsed), 60)
            self.lbl_elapsed.setText(f"{m}m {s}s")

        mode_text = update.derivative_mode.upper()
        if update.fallback_occurred:
            mode_text += " (FD fallback occurred)"
        self.lbl_derivative.setText(mode_text)

        if update.phase == "CANCELLED":
            self._on_cancelled()

    def _on_cancelled(self) -> None:
        self.progress_grp.setVisible(True)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_phase.setText("Phase: CANCELLED")
        self.lbl_status.setText("Cancelled by user")
        self.lbl_status.setStyleSheet("color: orange;")
        self.lbl_result_status.setText("CANCELLED")
        self.lbl_result_status.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: orange;"
        )

    def _on_finished(self, result: object) -> None:
        self._result = result
        self.progress_grp.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        from foster_eom.optimize.engine import OptimizationResult

        if isinstance(result, OptimizationResult):
            vm = OptimizeVM.from_result(result)
            self.lbl_result_status.setText(vm.status_label)

            color = {"FEASIBLE": "green", "NEAR-FEASIBLE": "#cc8800", "INFEASIBLE": "red"}
            self.lbl_result_status.setStyleSheet(
                f"font-weight: bold; font-size: 14px; color: {color.get(vm.status_label, 'black')};"
            )

            # Show provenance in status
            preset = self.preset_combo.currentText()
            spec = result.run_manifest
            self.lbl_status.setText(
                f"Done — {len(vm.candidates)} candidates | "
                f"Preset: {preset} | "
                f"Global evals: {spec.requested_global_budget:,} | "
                f"Seed: {spec.random_seed}"
            )
            self.lbl_status.setStyleSheet("")

            self.table.setRowCount(len(vm.candidates))
            for i, c in enumerate(vm.candidates):
                self.table.setItem(i, 0, QTableWidgetItem(str(c.rank)))
                self.table.setItem(i, 1, QTableWidgetItem(f"{c.objective:.6f}"))
                self.table.setItem(i, 2, QTableWidgetItem("✓" if c.feasible else "✗"))
                self.table.setItem(i, 3, QTableWidgetItem("✓" if c.near_feasible else "✗"))
                self.table.setItem(i, 4, QTableWidgetItem(c.numerical_status))
            if vm.candidates:
                self.table.selectRow(0)

        self.optimization_finished.emit(result)

    def _on_error(self, err_type: str, err_msg: str) -> None:
        self.progress_grp.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(f"Error: {err_type}")
        self.lbl_status.setStyleSheet("color: red;")
        self.lbl_result_status.setText("ERROR")
        self.lbl_result_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
        self.detail_text.setPlainText(err_msg)

    def _on_selection(self, row: int, _col: int = 0) -> None:
        from foster_eom.optimize.engine import OptimizationResult

        if not isinstance(self._result, OptimizationResult):
            return
        if 0 <= row < len(self._result.candidates):
            c = self._result.candidates[row]

            from foster_eom.gui.view_models.optimize_vm import (
                CandidateDetailVM,
                format_polish_provenance,
            )

            label_map = c.hard_constraint_labels or {}
            vm = CandidateDetailVM.from_candidate(row + 1, c, label_map=label_map)

            lines: list[str] = []

            # Header
            lines.append(f"Candidate #{vm.rank}")
            lines.append(f"Topology: {vm.topology_id}")
            lines.append(
                f"Objective: {vm.objective_base:.6f} (base) + {vm.objective_soft:.6f} (soft)"
            )
            lines.append(f"Feasible: {vm.feasible}  |  V_max: {vm.v_max:.6f}")
            lines.append(f"Numerical: {vm.numerical_status}")
            lines.append("")

            # Polish provenance
            polish_lines = format_polish_provenance(
                vm.local_polish_method, vm.local_polish_outcome
            )
            lines.extend(polish_lines)
            lines.append(f"Seed source: {vm.seed_source}")
            lines.append("")

            # Objective terms
            lines.append("Objective terms:")
            for k, v in vm.objective_terms.items():
                lines.append(f"  {k}: {v:.6f}")
            lines.append("")

            # Constraint summary
            lines.append(
                f"Hard constraints: {vm.total_hard} total | "
                f"{vm.violated_count} violated | v_max = {vm.v_max:.6f}"
            )
            lines.append("")

            # Violated constraints
            if vm.violated:
                lines.append("── Violated constraints (worst first) ──")
                for r in vm.violated:
                    lines.append(f"  {r.label}: margin = {r.margin:.6f}")
                lines.append("")
            else:
                lines.append("No hard-constraint violations.")
                lines.append("")

            # Closest active constraints
            if vm.closest_active:
                lines.append("── Closest active constraints ──")
                for r in vm.closest_active:
                    lines.append(f"  {r.label}: margin = {r.margin:.6f}")
                lines.append("")

            # Full list indicator
            remaining = vm.total_hard - vm.violated_count - len(vm.closest_active)
            if remaining > 0:
                lines.append(
                    f"[{remaining} additional constraints not shown — "
                    f"all with margin > {vm.closest_active[-1].margin:.4f}]"
                    if vm.closest_active
                    else f"[{remaining} additional constraints not shown]"
                )

            self.detail_text.setPlainText("\n".join(lines))

