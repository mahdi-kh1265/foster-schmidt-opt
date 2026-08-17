"""Project / Inputs page."""

from __future__ import annotations

import typing

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.state import ProjectState


class _FreqRow(QWidget):
    """Single frequency row with value + unit selector."""

    removed = Signal(object)

    def __init__(self, freq_hz: float = 10e6, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(4)
        self.spin.setRange(0.0001, 1e12)
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.unit_combo.setCurrentText("MHz")

        self._set_display(freq_hz)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))

        lay.addWidget(self.spin, 1)
        lay.addWidget(self.unit_combo)
        lay.addWidget(self.btn_remove)

        self.unit_combo.currentTextChanged.connect(self._on_unit_change)

    # ------------------------------------------------------------------
    _MULTIPLIERS: typing.ClassVar[dict[str, float]] = {
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
    }

    def _set_display(self, hz: float) -> None:
        unit = "MHz"
        for u in ("GHz", "MHz", "kHz", "Hz"):
            m = self._MULTIPLIERS[u]
            if hz >= m:
                unit = u
                break
        self.unit_combo.blockSignals(True)
        self.unit_combo.setCurrentText(unit)
        self.unit_combo.blockSignals(False)
        self.spin.setValue(hz / self._MULTIPLIERS[unit])

    def _on_unit_change(self) -> None:
        pass  # keep displayed number, user adjusts

    def freq_hz(self) -> float:
        return self.spin.value() * self._MULTIPLIERS[self.unit_combo.currentText()]


class ProjectPage(QWidget):
    """Full project-inputs page."""

    inputs_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        # --- Project Name ---
        name_grp = QGroupBox("Project")
        name_lay = QFormLayout(name_grp)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Untitled Project")
        name_lay.addRow("Name:", self.name_edit)
        lay.addWidget(name_grp)

        # --- Target Frequencies ---
        freq_grp = QGroupBox("Target Frequencies")
        freq_lay = QVBoxLayout(freq_grp)
        self.freq_list_layout = QVBoxLayout()
        freq_lay.addLayout(self.freq_list_layout)
        btn_add = QPushButton("+ Add Frequency")
        btn_add.clicked.connect(self._add_freq_row)
        freq_lay.addWidget(btn_add)
        lay.addWidget(freq_grp)

        # --- Sweep Band ---
        sweep_grp = QGroupBox("Verification Sweep Band")
        sweep_lay = QFormLayout(sweep_grp)
        self.sweep_min = QDoubleSpinBox()
        self.sweep_min.setDecimals(3)
        self.sweep_min.setRange(0.001, 1e6)
        self.sweep_min.setSuffix(" MHz")
        self.sweep_min.setValue(1.0)
        sweep_lay.addRow("Sweep Min:", self.sweep_min)
        self.sweep_max = QDoubleSpinBox()
        self.sweep_max.setDecimals(3)
        self.sweep_max.setRange(0.001, 1e6)
        self.sweep_max.setSuffix(" MHz")
        self.sweep_max.setValue(30.0)
        sweep_lay.addRow("Sweep Max:", self.sweep_max)
        lay.addWidget(sweep_grp)

        # --- Source ---
        src_grp = QGroupBox("RF Source")
        src_lay = QFormLayout(src_grp)
        self.source_mode = QComboBox()
        self.source_mode.addItems(["thevenin", "available_power", "generator_into_z0"])
        src_lay.addRow("Mode:", self.source_mode)
        self.vth_rms = QDoubleSpinBox()
        self.vth_rms.setDecimals(4)
        self.vth_rms.setRange(0.0, 1e6)
        self.vth_rms.setValue(1.0)
        self.vth_rms.setSuffix(" V rms")
        src_lay.addRow("V_th:", self.vth_rms)
        self.z_source = QDoubleSpinBox()
        self.z_source.setDecimals(2)
        self.z_source.setRange(0.01, 1e6)
        self.z_source.setValue(50.0)
        self.z_source.setSuffix(" Ω")
        src_lay.addRow("Z_source:", self.z_source)
        lay.addWidget(src_grp)

        # --- EOM ---
        eom_grp = QGroupBox("EOM Model")
        eom_lay = QFormLayout(eom_grp)
        self.eom_type = QComboBox()
        self.eom_type.addItems(
            [
                "ideal_capacitor",
                "lossy_capacitor",
                "mbvd",
                "tabular",
            ]
        )
        eom_lay.addRow("Type:", self.eom_type)
        self.eom_c0 = QDoubleSpinBox()
        self.eom_c0.setDecimals(2)
        self.eom_c0.setRange(0.01, 1e6)
        self.eom_c0.setValue(10.0)
        self.eom_c0.setSuffix(" pF")
        eom_lay.addRow("C₀:", self.eom_c0)
        self.eom_rs = QDoubleSpinBox()
        self.eom_rs.setDecimals(4)
        self.eom_rs.setRange(0.0, 1e6)
        self.eom_rs.setValue(0.0)
        self.eom_rs.setSuffix(" Ω")
        eom_lay.addRow("R_s:", self.eom_rs)
        self.eom_ls = QDoubleSpinBox()
        self.eom_ls.setDecimals(4)
        self.eom_ls.setRange(0.0, 1e6)
        self.eom_ls.setValue(0.0)
        self.eom_ls.setSuffix(" nH")
        eom_lay.addRow("L_s:", self.eom_ls)
        self.eom_g0 = QDoubleSpinBox()
        self.eom_g0.setDecimals(6)
        self.eom_g0.setRange(0.0, 1e6)
        self.eom_g0.setValue(0.0)
        self.eom_g0.setSuffix(" S")
        eom_lay.addRow("G₀:", self.eom_g0)

        self.eom_file_label = QLabel("(none)")
        eom_file_btn = QPushButton("Browse…")
        eom_file_btn.clicked.connect(self._browse_eom_file)
        eom_file_row = QHBoxLayout()
        eom_file_row.addWidget(self.eom_file_label, 1)
        eom_file_row.addWidget(eom_file_btn)
        eom_lay.addRow("Data file:", eom_file_row)

        lay.addWidget(eom_grp)

        # --- Topology ---
        topo_grp = QGroupBox("Topology / Pole Configuration")
        topo_lay = QFormLayout(topo_grp)
        self.n_branches = QSpinBox()
        self.n_branches.setRange(1, 2)
        self.n_branches.setValue(1)
        topo_lay.addRow("Branches:", self.n_branches)
        self.n_cells = QSpinBox()
        self.n_cells.setRange(1, 6)
        self.n_cells.setValue(1)
        topo_lay.addRow("Cells / branch:", self.n_cells)
        lay.addWidget(topo_grp)

        # --- Validation label ---
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red;")
        lay.addWidget(self.validation_label)

        lay.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ------------------------------------------------------------------
    def _add_freq_row(self, freq_hz: float = 10e6) -> None:
        row = _FreqRow(freq_hz)
        row.removed.connect(self._remove_freq_row)
        self.freq_list_layout.addWidget(row)

    def _remove_freq_row(self, row: _FreqRow) -> None:
        self.freq_list_layout.removeWidget(row)
        row.deleteLater()

    def _browse_eom_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EOM data file",
            "",
            "Touchstone (*.s1p);;CSV (*.csv);;All (*)",
        )
        if path:
            self.eom_file_label.setText(path)

    # ------------------------------------------------------------------
    def populate_from_state(self, state: ProjectState) -> None:
        """Load widgets from ProjectState."""
        self.name_edit.setText(state.name)

        # Clear existing freq rows
        while self.freq_list_layout.count():
            item = self.freq_list_layout.takeAt(0)
            if item is not None and item.widget():
                item.widget().deleteLater()

        for f in state.frequencies_hz:
            self._add_freq_row(f)
        if not state.frequencies_hz:
            self._add_freq_row(10e6)

        self.sweep_min.setValue(state.sweep_f_min_hz / 1e6)
        self.sweep_max.setValue(state.sweep_f_max_hz / 1e6)

        self.source_mode.setCurrentText(state.source.mode)
        self.vth_rms.setValue(state.source.vth_rms)
        self.z_source.setValue(state.source.z_source_ohm)

        self.eom_type.setCurrentText(state.eom.model_type)
        self.eom_c0.setValue((state.eom.c0_f or 0.0) * 1e12)
        self.eom_rs.setValue(state.eom.rs_ohm or 0.0)
        self.eom_ls.setValue((state.eom.ls_h or 0.0) * 1e9)
        self.eom_g0.setValue(state.eom.g0_s or 0.0)
        if state.eom.tabular_file:
            self.eom_file_label.setText(state.eom.tabular_file)

        self.n_branches.setValue(state.topology.n_branches)
        self.n_cells.setValue(state.topology.n_cells_per_branch)

    def write_to_state(self, state: ProjectState) -> None:
        """Write widget values back into ProjectState."""
        state.name = self.name_edit.text()

        freqs = []
        for i in range(self.freq_list_layout.count()):
            w = self.freq_list_layout.itemAt(i).widget()
            if isinstance(w, _FreqRow):
                freqs.append(w.freq_hz())
        state.frequencies_hz = freqs

        state.sweep_f_min_hz = self.sweep_min.value() * 1e6
        state.sweep_f_max_hz = self.sweep_max.value() * 1e6

        state.source.mode = self.source_mode.currentText()
        state.source.vth_rms = self.vth_rms.value()
        state.source.z_source_ohm = self.z_source.value()

        state.eom.model_type = self.eom_type.currentText()
        state.eom.c0_f = self.eom_c0.value() * 1e-12
        rs = self.eom_rs.value()
        state.eom.rs_ohm = rs if rs > 0 else None
        ls = self.eom_ls.value()
        state.eom.ls_h = (ls * 1e-9) if ls > 0 else None
        g0 = self.eom_g0.value()
        state.eom.g0_s = g0 if g0 > 0 else None

        txt = self.eom_file_label.text()
        state.eom.tabular_file = txt if txt != "(none)" else None

        state.topology.n_branches = self.n_branches.value()
        state.topology.n_cells_per_branch = self.n_cells.value()

    def validate(self) -> str | None:
        """Return error string or None if valid."""
        freqs = []
        for i in range(self.freq_list_layout.count()):
            w = self.freq_list_layout.itemAt(i).widget()
            if isinstance(w, _FreqRow):
                freqs.append(w.freq_hz())
        if not freqs:
            return "At least one target frequency is required."
        if self.sweep_min.value() >= self.sweep_max.value():
            return "Sweep min must be less than sweep max."
        f_min = self.sweep_min.value() * 1e6
        f_max = self.sweep_max.value() * 1e6
        for f in freqs:
            if f < f_min or f > f_max:
                return f"Target {f / 1e6:.3f} MHz outside sweep band [{f_min / 1e6:.3f}, {f_max / 1e6:.3f}] MHz."
        c0 = self.eom_c0.value()
        if c0 <= 0:
            return "C₀ must be positive."
        self.validation_label.setText("")
        return None
