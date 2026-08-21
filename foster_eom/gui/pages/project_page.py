"""Project / Inputs page."""

from __future__ import annotations

import typing

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    """Single frequency row with value + unit selector + optional voltage target."""

    removed = Signal(object)

    def __init__(self, freq_hz: float = 10e6, voltage_rms_v: float | None = None,
                 parent: QWidget | None = None):
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

        # Per-target voltage (optional)
        self.voltage_spin = QDoubleSpinBox()
        self.voltage_spin.setDecimals(3)
        self.voltage_spin.setRange(0.0, 1e6)
        self.voltage_spin.setValue(voltage_rms_v if voltage_rms_v is not None else 0.0)
        self.voltage_spin.setSuffix(" V rms")
        self.voltage_spin.setToolTip("Per-target EOM voltage (0 = no requirement)")
        self.voltage_spin.setFixedWidth(120)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))

        lay.addWidget(self.spin, 1)
        lay.addWidget(self.unit_combo)
        lay.addWidget(QLabel("V:"))
        lay.addWidget(self.voltage_spin)
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

    def voltage_target_rms_v(self) -> float | None:
        v = self.voltage_spin.value()
        return v if v > 0.0 else None


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
        freq_lay.addWidget(QLabel("Each target may optionally specify a required EOM voltage (0 = none)."))
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

        # === Matching Constraints ===
        match_grp = QGroupBox("Matching Constraints")
        match_lay = QFormLayout(match_grp)

        self.spin_gamma_max = QDoubleSpinBox()
        self.spin_gamma_max.setDecimals(3)
        self.spin_gamma_max.setRange(0.0, 1.0)
        self.spin_gamma_max.setSingleStep(0.01)
        self.spin_gamma_max.setValue(0.25)
        self.spin_gamma_max.setToolTip("Maximum reflection coefficient |Γ| at target frequencies")
        match_lay.addRow("Γ_max:", self.spin_gamma_max)

        self.spin_r_min = QDoubleSpinBox()
        self.spin_r_min.setDecimals(1)
        self.spin_r_min.setRange(0.1, 1e6)
        self.spin_r_min.setValue(35.0)
        self.spin_r_min.setSuffix(" Ω")
        self.spin_r_min.setToolTip("Minimum real part of Z_in")
        match_lay.addRow("R_in min:", self.spin_r_min)

        self.spin_r_max = QDoubleSpinBox()
        self.spin_r_max.setDecimals(1)
        self.spin_r_max.setRange(0.1, 1e6)
        self.spin_r_max.setValue(70.0)
        self.spin_r_max.setSuffix(" Ω")
        self.spin_r_max.setToolTip("Maximum real part of Z_in")
        match_lay.addRow("R_in max:", self.spin_r_max)

        self.spin_x_max = QDoubleSpinBox()
        self.spin_x_max.setDecimals(1)
        self.spin_x_max.setRange(0.0, 1e6)
        self.spin_x_max.setValue(20.0)
        self.spin_x_max.setSuffix(" Ω")
        self.spin_x_max.setToolTip("Maximum |Im(Z_in)|")
        match_lay.addRow("|X_in| max:", self.spin_x_max)

        lay.addWidget(match_grp)

        # === Component Limits ===
        comp_grp = QGroupBox("Component Limits")
        comp_lay = QFormLayout(comp_grp)

        # L min
        l_min_row = QHBoxLayout()
        self.spin_l_min = QDoubleSpinBox()
        self.spin_l_min.setDecimals(2)
        self.spin_l_min.setRange(0.01, 1e9)
        self.spin_l_min.setValue(10.0)
        self.combo_l_min_unit = QComboBox()
        self.combo_l_min_unit.addItems(["nH", "µH", "mH"])
        self.combo_l_min_unit.setCurrentText("nH")
        l_min_row.addWidget(self.spin_l_min, 1)
        l_min_row.addWidget(self.combo_l_min_unit)
        comp_lay.addRow("L min:", l_min_row)

        # L max
        l_max_row = QHBoxLayout()
        self.spin_l_max = QDoubleSpinBox()
        self.spin_l_max.setDecimals(2)
        self.spin_l_max.setRange(0.01, 1e9)
        self.spin_l_max.setValue(100.0)
        self.combo_l_max_unit = QComboBox()
        self.combo_l_max_unit.addItems(["nH", "µH", "mH"])
        self.combo_l_max_unit.setCurrentText("µH")
        l_max_row.addWidget(self.spin_l_max, 1)
        l_max_row.addWidget(self.combo_l_max_unit)
        comp_lay.addRow("L max:", l_max_row)

        # C min
        c_min_row = QHBoxLayout()
        self.spin_c_min = QDoubleSpinBox()
        self.spin_c_min.setDecimals(2)
        self.spin_c_min.setRange(0.001, 1e9)
        self.spin_c_min.setValue(0.2)
        self.combo_c_min_unit = QComboBox()
        self.combo_c_min_unit.addItems(["pF", "nF", "µF"])
        self.combo_c_min_unit.setCurrentText("pF")
        c_min_row.addWidget(self.spin_c_min, 1)
        c_min_row.addWidget(self.combo_c_min_unit)
        comp_lay.addRow("C min:", c_min_row)

        # C max
        c_max_row = QHBoxLayout()
        self.spin_c_max = QDoubleSpinBox()
        self.spin_c_max.setDecimals(2)
        self.spin_c_max.setRange(0.001, 1e9)
        self.spin_c_max.setValue(20.0)
        self.combo_c_max_unit = QComboBox()
        self.combo_c_max_unit.addItems(["pF", "nF", "µF"])
        self.combo_c_max_unit.setCurrentText("nF")
        c_max_row.addWidget(self.spin_c_max, 1)
        c_max_row.addWidget(self.combo_c_max_unit)
        comp_lay.addRow("C max:", c_max_row)

        lay.addWidget(comp_grp)

        # === Stress Limits (Primary) ===
        stress_grp = QGroupBox("Stress Limits")
        stress_lay = QFormLayout(stress_grp)

        self.spin_i_src_max = QDoubleSpinBox()
        self.spin_i_src_max.setDecimals(3)
        self.spin_i_src_max.setRange(0.001, 1e6)
        self.spin_i_src_max.setValue(0.5)
        self.spin_i_src_max.setSuffix(" A rms")
        self.spin_i_src_max.setToolTip("Maximum RMS source current")
        stress_lay.addRow("I_source max:", self.spin_i_src_max)

        self.spin_v_off_target = QDoubleSpinBox()
        self.spin_v_off_target.setDecimals(1)
        self.spin_v_off_target.setRange(0.1, 1e6)
        self.spin_v_off_target.setValue(50.0)
        self.spin_v_off_target.setSuffix(" V rms")
        self.spin_v_off_target.setToolTip("Maximum EOM RMS voltage at off-target frequencies")
        stress_lay.addRow("V_EOM off-target max:", self.spin_v_off_target)

        lay.addWidget(stress_grp)

        # === Advanced (Collapsible) ===
        self.advanced_toggle = QCheckBox("Show Advanced Settings")
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        lay.addWidget(self.advanced_toggle)

        self.advanced_grp = QGroupBox("Advanced Objective Weights & Stress")
        adv_lay = QFormLayout(self.advanced_grp)

        self.spin_w_gamma = QDoubleSpinBox()
        self.spin_w_gamma.setDecimals(3)
        self.spin_w_gamma.setRange(0.0, 100.0)
        self.spin_w_gamma.setValue(1.0)
        self.spin_w_gamma.setToolTip("Weight for reflection coefficient Γ in objective function")
        adv_lay.addRow("γ weight:", self.spin_w_gamma)

        self.spin_w_voltage = QDoubleSpinBox()
        self.spin_w_voltage.setDecimals(3)
        self.spin_w_voltage.setRange(0.0, 100.0)
        self.spin_w_voltage.setValue(1.0)
        self.spin_w_voltage.setToolTip("Weight for voltage matching in objective function")
        adv_lay.addRow("Voltage weight:", self.spin_w_voltage)

        self.spin_w_loss = QDoubleSpinBox()
        self.spin_w_loss.setDecimals(3)
        self.spin_w_loss.setRange(0.0, 100.0)
        self.spin_w_loss.setValue(0.0)
        self.spin_w_loss.setToolTip("Weight for insertion loss in objective function")
        adv_lay.addRow("Loss weight:", self.spin_w_loss)

        self.spin_w_complexity = QDoubleSpinBox()
        self.spin_w_complexity.setDecimals(3)
        self.spin_w_complexity.setRange(0.0, 100.0)
        self.spin_w_complexity.setValue(0.0)
        self.spin_w_complexity.setToolTip("Weight for circuit complexity in objective function")
        adv_lay.addRow("Complexity weight:", self.spin_w_complexity)

        self.spin_cap_v_stress = QDoubleSpinBox()
        self.spin_cap_v_stress.setDecimals(1)
        self.spin_cap_v_stress.setRange(1.0, 1e6)
        self.spin_cap_v_stress.setValue(100.0)
        self.spin_cap_v_stress.setSuffix(" V")
        self.spin_cap_v_stress.setToolTip("Default peak voltage limit for capacitors")
        adv_lay.addRow("Cap peak V limit:", self.spin_cap_v_stress)

        self.spin_ind_i_stress = QDoubleSpinBox()
        self.spin_ind_i_stress.setDecimals(3)
        self.spin_ind_i_stress.setRange(0.001, 1e6)
        self.spin_ind_i_stress.setValue(1.0)
        self.spin_ind_i_stress.setSuffix(" A")
        self.spin_ind_i_stress.setToolTip("Default peak current limit for inductors")
        adv_lay.addRow("Ind peak I limit:", self.spin_ind_i_stress)

        self.advanced_grp.setVisible(False)
        lay.addWidget(self.advanced_grp)

        # --- Validation label ---
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red;")
        lay.addWidget(self.validation_label)

        lay.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------
    _L_UNITS: typing.ClassVar[dict[str, float]] = {"nH": 1e-9, "µH": 1e-6, "mH": 1e-3}
    _C_UNITS: typing.ClassVar[dict[str, float]] = {"pF": 1e-12, "nF": 1e-9, "µF": 1e-6}

    def _l_to_h(self, spin: QDoubleSpinBox, combo: QComboBox) -> float:
        return spin.value() * self._L_UNITS[combo.currentText()]

    def _c_to_f(self, spin: QDoubleSpinBox, combo: QComboBox) -> float:
        return spin.value() * self._C_UNITS[combo.currentText()]

    def _set_l_display(self, h: float, spin: QDoubleSpinBox, combo: QComboBox) -> None:
        """Set L spin+combo from a value in henries."""
        for unit, mult in [("mH", 1e-3), ("µH", 1e-6), ("nH", 1e-9)]:
            if h >= mult:
                combo.setCurrentText(unit)
                spin.setValue(h / mult)
                return
        combo.setCurrentText("nH")
        spin.setValue(h / 1e-9)

    def _set_c_display(self, f: float, spin: QDoubleSpinBox, combo: QComboBox) -> None:
        """Set C spin+combo from a value in farads."""
        for unit, mult in [("µF", 1e-6), ("nF", 1e-9), ("pF", 1e-12)]:
            if f >= mult:
                combo.setCurrentText(unit)
                spin.setValue(f / mult)
                return
        combo.setCurrentText("pF")
        spin.setValue(f / 1e-12)

    # ------------------------------------------------------------------
    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_grp.setVisible(checked)

    def _add_freq_row(self, freq_hz: float = 10e6,
                      voltage_rms_v: float | None = None) -> None:
        row = _FreqRow(freq_hz, voltage_rms_v)
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

        # Per-target voltage: zip with voltage_targets_rms_v if available
        voltages = getattr(state, "voltage_targets_rms_v", [])
        for i, f in enumerate(state.frequencies_hz):
            v = voltages[i] if i < len(voltages) else None
            self._add_freq_row(f, v)
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

        # Matching constraints
        mp = state.match_params
        self.spin_gamma_max.setValue(mp.gamma_max)
        self.spin_r_min.setValue(mp.resistance_min_ohm)
        self.spin_r_max.setValue(mp.resistance_max_ohm)
        self.spin_x_max.setValue(mp.max_abs_reactance_ohm)

        # Component limits
        cl = state.component_limits
        self._set_l_display(cl.l_min_h, self.spin_l_min, self.combo_l_min_unit)
        self._set_l_display(cl.l_max_h, self.spin_l_max, self.combo_l_max_unit)
        self._set_c_display(cl.c_min_f, self.spin_c_min, self.combo_c_min_unit)
        self._set_c_display(cl.c_max_f, self.spin_c_max, self.combo_c_max_unit)

        # Stress limits
        sp = state.stress_params
        self.spin_i_src_max.setValue(sp.source_current_rms_max_a)
        self.spin_v_off_target.setValue(sp.off_target_eom_peak_rms_v)

        # Advanced
        ow = state.objective_weights
        self.spin_w_gamma.setValue(ow.weight_gamma)
        self.spin_w_voltage.setValue(ow.weight_voltage)
        self.spin_w_loss.setValue(ow.weight_loss)
        self.spin_w_complexity.setValue(ow.weight_complexity)
        self.spin_cap_v_stress.setValue(sp.default_cap_peak_voltage_v)
        self.spin_ind_i_stress.setValue(sp.default_ind_peak_current_a)

    def write_to_state(self, state: ProjectState) -> None:
        """Write widget values back into ProjectState."""
        state.name = self.name_edit.text()

        freqs = []
        voltages = []
        for i in range(self.freq_list_layout.count()):
            w = self.freq_list_layout.itemAt(i).widget()
            if isinstance(w, _FreqRow):
                freqs.append(w.freq_hz())
                voltages.append(w.voltage_target_rms_v())
        state.frequencies_hz = freqs
        state.voltage_targets_rms_v = voltages  # type: ignore[attr-defined]

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

        # Matching constraints
        state.match_params.gamma_max = self.spin_gamma_max.value()
        state.match_params.resistance_min_ohm = self.spin_r_min.value()
        state.match_params.resistance_max_ohm = self.spin_r_max.value()
        state.match_params.max_abs_reactance_ohm = self.spin_x_max.value()

        # Component limits (with unit conversion)
        state.component_limits.l_min_h = self._l_to_h(self.spin_l_min, self.combo_l_min_unit)
        state.component_limits.l_max_h = self._l_to_h(self.spin_l_max, self.combo_l_max_unit)
        state.component_limits.c_min_f = self._c_to_f(self.spin_c_min, self.combo_c_min_unit)
        state.component_limits.c_max_f = self._c_to_f(self.spin_c_max, self.combo_c_max_unit)

        # Stress limits
        state.stress_params.source_current_rms_max_a = self.spin_i_src_max.value()
        state.stress_params.off_target_eom_peak_rms_v = self.spin_v_off_target.value()

        # Advanced
        state.objective_weights.weight_gamma = self.spin_w_gamma.value()
        state.objective_weights.weight_voltage = self.spin_w_voltage.value()
        state.objective_weights.weight_loss = self.spin_w_loss.value()
        state.objective_weights.weight_complexity = self.spin_w_complexity.value()
        state.stress_params.default_cap_peak_voltage_v = self.spin_cap_v_stress.value()
        state.stress_params.default_ind_peak_current_a = self.spin_ind_i_stress.value()

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
        # Matching constraint validation: R_min <= R_max
        if self.spin_r_min.value() > self.spin_r_max.value():
            return "R_in min must be ≤ R_in max."
        # Component limit validation: L_min <= L_max, C_min <= C_max
        l_min = self._l_to_h(self.spin_l_min, self.combo_l_min_unit)
        l_max = self._l_to_h(self.spin_l_max, self.combo_l_max_unit)
        if l_min > l_max:
            return "L min must be ≤ L max."
        c_min = self._c_to_f(self.spin_c_min, self.combo_c_min_unit)
        c_max = self._c_to_f(self.spin_c_max, self.combo_c_max_unit)
        if c_min > c_max:
            return "C min must be ≤ C max."
        self.validation_label.setText("")
        return None
