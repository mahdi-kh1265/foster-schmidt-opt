"""Verify / P06 page with matplotlib plots."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.controllers.verify_ctrl import VerifyCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.gui.view_models.verify_vm import VerifyVM
from foster_eom.gui.workers.base_worker import BaseWorker

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _MPL_OK = True
except ImportError:
    _MPL_OK = False


class VerifyPage(QWidget):
    """P06 verification page with plots and tables."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: ProjectState | None = None
        self._opt_result = None
        self._sweep_result = None
        self._worker: BaseWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Controls ---
        ctrl = QHBoxLayout()
        self.btn_run = QPushButton("Run Verification (P06)")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)
        ctrl.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress, 1)
        self.lbl_status = QLabel("Idle — requires optimization result")
        ctrl.addWidget(self.lbl_status)
        lay.addLayout(ctrl)

        # --- Tabs: Plots / Tables ---
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot tabs
        self.plot_tabs = QTabWidget()
        if _MPL_OK:
            # Z_in plot
            self.fig_zin = Figure(figsize=(6, 3), dpi=100)
            self.canvas_zin = FigureCanvasQTAgg(self.fig_zin)
            self.plot_tabs.addTab(self.canvas_zin, "Z_in")

            # Match / Gamma plot
            self.fig_gamma = Figure(figsize=(6, 3), dpi=100)
            self.canvas_gamma = FigureCanvasQTAgg(self.fig_gamma)
            self.plot_tabs.addTab(self.canvas_gamma, "Match (Γ)")

            # EOM voltage plot
            self.fig_veom = Figure(figsize=(6, 3), dpi=100)
            self.canvas_veom = FigureCanvasQTAgg(self.fig_veom)
            self.plot_tabs.addTab(self.canvas_veom, "EOM Voltage")
        else:
            self.plot_tabs.addTab(QLabel("matplotlib not available"), "Plots")

        splitter.addWidget(self.plot_tabs)

        # Tables
        table_tabs = QTabWidget()

        # Q / Resonance table
        self.q_table = QTableWidget()
        self.q_table.setColumnCount(3)
        self.q_table.setHorizontalHeaderLabels(["f₀ (MHz)", "Q (3dB)", "Z_peak (Ω)"])
        self.q_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.q_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.q_table.setAlternatingRowColors(True)
        table_tabs.addTab(self.q_table, "Resonance / Q")

        # Stress table
        self.stress_table = QTableWidget()
        self.stress_table.setColumnCount(5)
        self.stress_table.setHorizontalHeaderLabels(
            [
                "Element",
                "V_peak (V)",
                "I_peak (A)",
                "P_diss (W)",
                "Freq (MHz)",
            ]
        )
        self.stress_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stress_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stress_table.setAlternatingRowColors(True)
        table_tabs.addTab(self.stress_table, "Stress")

        # Warnings
        self.warn_label = QLabel("No warnings.")
        table_tabs.addTab(self.warn_label, "Warnings")

        splitter.addWidget(table_tabs)
        lay.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    def set_prereqs(self, state: ProjectState, opt_result: object) -> None:
        self._state = state
        self._opt_result = opt_result
        enabled = opt_result is not None
        self.btn_run.setEnabled(enabled)
        if enabled:
            self.lbl_status.setText("Ready")
        else:
            self.lbl_status.setText("Idle — requires optimization result")

    def set_stale(self) -> None:
        self.lbl_status.setText("⚠ Results stale")
        self.lbl_status.setStyleSheet("color: orange;")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self._state or not self._opt_result:
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running…")
        self.lbl_status.setStyleSheet("")

        state = self._state
        opt = self._opt_result

        worker = BaseWorker(
            fn=lambda: VerifyCtrl.run(state, opt),
            project_revision=state.revision,
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self, result: object) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

        try:
            sweep_res, q_metrics, stress_res = result
            self._sweep_result = sweep_res
            vm = VerifyVM.from_results(q_metrics, stress_res)
            self.lbl_status.setText("Done")
            self.lbl_status.setStyleSheet("color: green;")

            # Populate Q table
            self.q_table.setRowCount(len(vm.q_metrics))
            for i, q in enumerate(vm.q_metrics):
                self.q_table.setItem(i, 0, QTableWidgetItem(f"{q.f0_hz / 1e6:.4f}"))
                self.q_table.setItem(i, 1, QTableWidgetItem(f"{q.q_3db:.2f}"))
                self.q_table.setItem(i, 2, QTableWidgetItem(f"{q.z_peak_ohm:.2f}"))

            # Populate stress table
            self.stress_table.setRowCount(len(vm.stress))
            for i, s in enumerate(vm.stress):
                self.stress_table.setItem(i, 0, QTableWidgetItem(s.element))
                self.stress_table.setItem(i, 1, QTableWidgetItem(f"{s.v_peak:.4f}"))
                self.stress_table.setItem(i, 2, QTableWidgetItem(f"{s.i_peak:.6f}"))
                self.stress_table.setItem(i, 3, QTableWidgetItem(f"{s.p_diss_w:.6f}"))
                self.stress_table.setItem(i, 4, QTableWidgetItem(f"{s.freq_hz / 1e6:.4f}"))

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.lbl_status.setText("FAILED")
            self.lbl_status.setStyleSheet("color: red;")
            self.warn_label.setText(f"Rendering error: {e}")
            print(f"VerifyPage rendering error:\n{err_msg}")
            return

        # Update plots
        if _MPL_OK and sweep_res is not None:
            self._plot_sweep(sweep_res)

    def _on_error(self, err_type: str, err_msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Error: {err_type}")
        self.lbl_status.setStyleSheet("color: red;")
        self.warn_label.setText(err_msg)

    def _plot_sweep(self, sweep) -> None:
        """Plot impedance, match, and EOM voltage from sweep result."""
        try:
            freqs_mhz = [f / 1e6 for f in sweep.frequencies_hz]
            z_in = sweep.z_in

            # Z_in: magnitude and phase
            self.fig_zin.clear()
            ax1, ax2 = self.fig_zin.subplots(1, 2)
            ax1.semilogy(freqs_mhz, [abs(z) for z in z_in])
            ax1.set_xlabel("Frequency (MHz)")
            ax1.set_ylabel("|Z_in| (Ω)")
            ax1.set_title("Input Impedance Magnitude")
            ax1.grid(True, alpha=0.3)

            import cmath

            ax2.plot(freqs_mhz, [cmath.phase(z) * 180 / 3.14159 for z in z_in])
            ax2.set_xlabel("Frequency (MHz)")
            ax2.set_ylabel("∠Z_in (°)")
            ax2.set_title("Input Impedance Phase")
            ax2.grid(True, alpha=0.3)
            self.fig_zin.tight_layout()
            self.canvas_zin.draw()

            # Gamma
            self.fig_gamma.clear()
            ax_g = self.fig_gamma.add_subplot(111)
            if hasattr(sweep, "gamma"):
                ax_g.plot(freqs_mhz, [abs(g) for g in sweep.gamma])
                ax_g.set_ylabel("|Γ|")
            elif hasattr(sweep, "reflection_coefficient"):
                ax_g.plot(freqs_mhz, [abs(g) for g in sweep.reflection_coefficient])
                ax_g.set_ylabel("|Γ|")
            else:
                z_ref = 50.0
                gamma = [(z - z_ref) / (z + z_ref) for z in z_in]
                ax_g.plot(freqs_mhz, [abs(g) for g in gamma])
                ax_g.set_ylabel("|Γ|")
            ax_g.set_xlabel("Frequency (MHz)")
            ax_g.set_title("Reflection Coefficient")
            ax_g.grid(True, alpha=0.3)
            self.fig_gamma.tight_layout()
            self.canvas_gamma.draw()

            # EOM voltage
            self.fig_veom.clear()
            ax_v = self.fig_veom.add_subplot(111)
            if hasattr(sweep, "v_eom_rms"):
                ax_v.plot(freqs_mhz, sweep.v_eom_rms)
                ax_v.set_ylabel("V_EOM (V rms)")
            elif hasattr(sweep, "eom_voltage"):
                ax_v.plot(freqs_mhz, [abs(v) for v in sweep.eom_voltage])
                ax_v.set_ylabel("|V_EOM| (V)")
            ax_v.set_xlabel("Frequency (MHz)")
            ax_v.set_title("EOM Voltage vs Frequency")
            ax_v.grid(True, alpha=0.3)
            self.fig_veom.tight_layout()
            self.canvas_veom.draw()
        except Exception:
            pass  # degrade gracefully if sweep format unexpected
