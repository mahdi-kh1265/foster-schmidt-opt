"""SPICE Export / P11 page."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.controllers.spice_ctrl import SpiceCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.gui.workers.base_worker import BaseWorker


class SpicePage(QWidget):
    """P11 SPICE export and validation page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._real_result = None
        self._netlist: str | None = None
        self._worker: BaseWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Controls ---
        ctrl = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Netlist")
        self.btn_generate.clicked.connect(self._generate)
        self.btn_generate.setEnabled(False)
        ctrl.addWidget(self.btn_generate)

        self.btn_export = QPushButton("Export .cir…")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setEnabled(False)
        ctrl.addWidget(self.btn_export)

        self.btn_validate = QPushButton("Run ngspice Validation")
        self.btn_validate.clicked.connect(self._validate)
        self.btn_validate.setEnabled(False)
        ctrl.addWidget(self.btn_validate)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress, 1)

        self.lbl_status = QLabel("Idle — requires realization")
        ctrl.addWidget(self.lbl_status)
        lay.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Netlist viewer ---
        nl_grp = QGroupBox("Generated Netlist")
        nl_lay = QVBoxLayout(nl_grp)
        self.netlist_edit = QTextEdit()
        self.netlist_edit.setReadOnly(True)
        self.netlist_edit.setFontFamily("Consolas")
        nl_lay.addWidget(self.netlist_edit)
        splitter.addWidget(nl_grp)

        # --- Validation result ---
        val_grp = QGroupBox("MNA vs SPICE Comparison")
        val_lay = QVBoxLayout(val_grp)
        self.lbl_val_status = QLabel("—")
        self.lbl_val_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        val_lay.addWidget(self.lbl_val_status)

        self.val_table = QTableWidget()
        self.val_table.setColumnCount(4)
        self.val_table.setHorizontalHeaderLabels(
            [
                "Quantity",
                "Max Rel Error",
                "Max Phase (°)",
                "Status",
            ]
        )
        self.val_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.val_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.val_table.setAlternatingRowColors(True)
        val_lay.addWidget(self.val_table)

        self.lbl_fail_reason = QLabel("")
        self.lbl_fail_reason.setWordWrap(True)
        val_lay.addWidget(self.lbl_fail_reason)

        splitter.addWidget(val_grp)
        lay.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    def set_prereqs(self, state: ProjectState, real_result: object) -> None:
        self._state = state
        self._real_result = real_result
        enabled = real_result is not None
        self.btn_generate.setEnabled(enabled)
        self.btn_validate.setEnabled(enabled)
        if enabled:
            self.lbl_status.setText("Ready")
        else:
            self.lbl_status.setText("Idle — requires realization")

    def set_stale(self) -> None:
        self.lbl_status.setText("⚠ Results stale")
        self.lbl_status.setStyleSheet("color: orange;")

    # ------------------------------------------------------------------
    def _generate(self) -> None:
        if not self._state or not self._real_result:
            return
        try:
            self._netlist = SpiceCtrl.export_netlist(self._state, self._real_result)
            self.netlist_edit.setPlainText(str(self._netlist))
            self.btn_export.setEnabled(True)
            self.lbl_status.setText("Netlist generated")
            self.lbl_status.setStyleSheet("color: green;")
        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")
            self.lbl_status.setStyleSheet("color: red;")

    def _export(self) -> None:
        if not self._netlist:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Netlist",
            "",
            "SPICE Netlist (*.cir);;All (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(self._netlist))
            self.lbl_status.setText(f"Exported to {path}")

    def _validate(self) -> None:
        if not self._state or not self._real_result:
            return
        self.btn_validate.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running ngspice validation…")
        self.lbl_status.setStyleSheet("")
        self.val_table.setRowCount(0)
        self.lbl_val_status.setText("—")
        self.lbl_fail_reason.setText("")

        state = self._state
        real = self._real_result

        worker = BaseWorker(
            fn=lambda: SpiceCtrl.validate(state, real),
            project_revision=state.revision,
        )
        worker.signals.finished.connect(self._on_val_finished)
        worker.signals.error.connect(self._on_val_error)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_val_finished(self, result: object) -> None:
        self.progress.setVisible(False)
        self.btn_validate.setEnabled(True)
        self.lbl_status.setText("Validation complete")
        self.lbl_status.setStyleSheet("color: green;")

        from foster_eom.gui.view_models.spice_vm import SpiceVM

        vm = SpiceVM.from_report(result)

        status_colors = {
            "PASS": "green",
            "WARN": "#cc8800",
            "FAIL": "red",
            "UNSUPPORTED": "gray",
            "SOLVER UNAVAILABLE": "gray",
        }
        self.lbl_val_status.setText(vm.status_label)
        self.lbl_val_status.setStyleSheet(
            f"font-weight: bold; font-size: 14px; "
            f"color: {status_colors.get(vm.status_label, 'black')};"
        )

        if vm.fail_reason:
            self.lbl_fail_reason.setText(f"Reason: {vm.fail_reason}")

        self.val_table.setRowCount(len(vm.comparisons))
        for i, c in enumerate(vm.comparisons):
            self.val_table.setItem(i, 0, QTableWidgetItem(c.quantity))
            self.val_table.setItem(i, 1, QTableWidgetItem(f"{c.max_rel_err:.2e}"))
            self.val_table.setItem(i, 2, QTableWidgetItem(f"{c.max_phase_deg:.2f}"))

            status_item = QTableWidgetItem(c.status)
            color_map = {
                "PASS": Qt.GlobalColor.darkGreen,
                "WARN": Qt.GlobalColor.darkYellow,
                "FAIL": Qt.GlobalColor.red,
            }
            if c.status in color_map:
                status_item.setForeground(color_map[c.status])
            self.val_table.setItem(i, 3, status_item)

    def _on_val_error(self, err_type: str, err_msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_validate.setEnabled(True)

        if "ngspice" in err_msg.lower() or "not found" in err_msg.lower():
            self.lbl_val_status.setText("SOLVER UNAVAILABLE")
            self.lbl_val_status.setStyleSheet("font-weight: bold; font-size: 14px; color: gray;")
            self.lbl_status.setText("ngspice not found")
            self.lbl_status.setStyleSheet("color: gray;")
        else:
            self.lbl_val_status.setText("ERROR")
            self.lbl_val_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
            self.lbl_status.setText(f"Error: {err_type}")
            self.lbl_status.setStyleSheet("color: red;")
        self.lbl_fail_reason.setText(err_msg)
