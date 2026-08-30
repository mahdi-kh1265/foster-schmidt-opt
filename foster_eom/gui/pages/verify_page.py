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


class PlotCursor:
    """Interactive engineering cursor for Matplotlib axes."""
    def __init__(self, canvas, axes, freqs_mhz, formatters):
        self.canvas = canvas
        self.axes = axes if isinstance(axes, list) else [axes]
        import numpy as np
        self.freqs_mhz = np.array(freqs_mhz)
        self.formatters = formatters

        self.hover_lines = [ax.axvline(x=0, color='gray', linestyle=':', visible=False, zorder=90) for ax in self.axes]
        self.pinned_lines = [ax.axvline(x=0, color='blue', linestyle='--', visible=False, zorder=90) for ax in self.axes]

        ax = self.axes[0]
        self.hover_text = ax.text(0.05, 0.95, "", transform=ax.transAxes,
                                  va='top', ha='left',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                                  visible=False, zorder=100)
        self.pinned_text = ax.text(0.95, 0.95, "", transform=ax.transAxes,
                                   va='top', ha='right',
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                                   visible=False, zorder=100)

        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)

    def find_nearest_idx(self, x):
        import numpy as np
        return (np.abs(self.freqs_mhz - x)).argmin()

    def get_text(self, idx):
        return "\n".join(f(idx) for f in self.formatters)

    def on_hover(self, event):
        if event.inaxes not in self.axes:
            for hl in self.hover_lines:
                hl.set_visible(False)
            self.hover_text.set_visible(False)
            self.canvas.draw_idle()
            return

        idx = self.find_nearest_idx(event.xdata)
        x_val = self.freqs_mhz[idx]

        for hl in self.hover_lines:
            hl.set_xdata([x_val, x_val])
            hl.set_visible(True)

        self.hover_text.set_text(self.get_text(idx))
        self.hover_text.set_visible(True)
        self.canvas.draw_idle()

    def on_click(self, event):
        if getattr(event, 'button', None) == 3:
            self.clear_pinned()
            return

        if getattr(event, 'inaxes', None) not in self.axes:
            return

        if getattr(event, 'button', None) == 1:
            idx = self.find_nearest_idx(event.xdata)
            x_val = self.freqs_mhz[idx]

            for pl in self.pinned_lines:
                pl.set_xdata([x_val, x_val])
                pl.set_visible(True)

            self.pinned_text.set_text(self.get_text(idx))
            self.pinned_text.set_visible(True)
            self.canvas.draw_idle()

    def on_key(self, event):
        if getattr(event, 'key', None) == 'escape':
            self.clear_pinned()

    def clear_pinned(self):
        for pl in self.pinned_lines:
            pl.set_visible(False)
        self.pinned_text.set_visible(False)
        self.canvas.draw_idle()


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
        self.q_table.setColumnCount(5)
        self.q_table.setHorizontalHeaderLabels(["Target (MHz)", "f₀ (MHz)", "Q (3dB)", "Usable BW (MHz)", "Status"])
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
            sweep_res, q_metrics, stress_res, z_in_sweep = result
            self._sweep_result = sweep_res
            vm = VerifyVM.from_results(q_metrics, stress_res)
            self.lbl_status.setText("Done")
            self.lbl_status.setStyleSheet("color: green;")

            # Populate Q table
            self.q_table.setRowCount(len(vm.q_metrics))
            for i, q in enumerate(vm.q_metrics):
                self.q_table.setItem(i, 0, QTableWidgetItem(f"{q.target_hz / 1e6:.4f}"))
                self.q_table.setItem(i, 1, QTableWidgetItem(f"{q.f0_hz / 1e6:.4f}" if q.f0_hz == q.f0_hz else "N/A"))
                self.q_table.setItem(i, 2, QTableWidgetItem(f"{q.q_3db:.2f}" if q.q_3db == q.q_3db else "N/A"))
                self.q_table.setItem(i, 3, QTableWidgetItem(f"{q.usable_bandwidth_hz / 1e6:.4f}" if q.usable_bandwidth_hz == q.usable_bandwidth_hz else "N/A"))
                self.q_table.setItem(i, 4, QTableWidgetItem(q.status))

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
        if _MPL_OK and self._sweep_result is not None:
            self._plot_sweep(self._sweep_result, z_in_sweep, vm.q_metrics)

    def _on_error(self, err_type: str, err_msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Error: {err_type}")
        self.lbl_status.setStyleSheet("color: red;")
        self.warn_label.setText(err_msg)

    def _plot_sweep(self, sweep, z_in_sweep: list[complex | None], q_metrics: list = []) -> None:
        """Plot impedance, match, and EOM voltage from sweep result."""
        try:
            freqs_mhz = [f / 1e6 for f in sweep.frequencies_hz]

            import cmath
            import math
            valid_z = [(f, z) for f, z in zip(freqs_mhz, z_in_sweep, strict=True) if z is not None]

            targets = []
            if self._state:
                for f_hz, v_rms in zip(self._state.frequencies_hz, self._state.voltage_targets_rms_v):
                    targets.append({"freq_hz": f_hz, "v_rms": v_rms})
            gamma_max = self._state.match_params.gamma_max if self._state else None

            v_lines = []
            for t in targets:
                v_lines.append((t['freq_hz'] / 1e6, 'green', '--', 0.5))
            for q in q_metrics:
                if getattr(q, 'f0_hz', None):
                    v_lines.append((q.f0_hz / 1e6, 'purple', ':', 0.7))
                if getattr(q, 'f_low_hz', None):
                    v_lines.append((q.f_low_hz / 1e6, 'purple', ':', 0.7))
                if getattr(q, 'f_high_hz', None):
                    v_lines.append((q.f_high_hz / 1e6, 'purple', ':', 0.7))

            def add_overlays(ax, is_voltage=False, is_gamma=False):
                for x, color, ls, alpha in v_lines:
                    ax.axvline(x, color=color, linestyle=ls, alpha=alpha, zorder=5)
                if is_gamma and gamma_max is not None:
                    ax.axhline(gamma_max, color='red', linestyle='--', alpha=0.5, zorder=5)
                if is_voltage:
                    for t in targets:
                        if t['v_rms'] is not None:
                            ax.axhline(t['v_rms'], color='red', linestyle='--', alpha=0.5, zorder=5)

            self._cursors = []

            self.fig_zin.clear()
            ax1, ax2 = self.fig_zin.subplots(1, 2)
            if valid_z:
                f_z = [x[0] for x in valid_z]
                z_vals = [x[1] for x in valid_z]
                ax1.semilogy(f_z, [abs(z) for z in z_vals])
                ax1.set_xlabel("Frequency (MHz)")
                ax1.set_ylabel("|Z_in| (Ω)")
                ax1.set_title("Input Impedance Magnitude")
                ax1.grid(True, alpha=0.3)
                add_overlays(ax1)

                ax2.plot(f_z, [cmath.phase(z) * 180 / math.pi for z in z_vals])
                ax2.set_xlabel("Frequency (MHz)")
                ax2.set_ylabel("∠Z_in (°)")
                ax2.set_title("Input Impedance Phase")
                ax2.grid(True, alpha=0.3)
                add_overlays(ax2)

                def fmt_zin(idx):
                    z = z_vals[idx]
                    return f"f = {f_z[idx]:.4f} MHz\n|Z_in| = {abs(z):.2f} Ω\nphase = {cmath.phase(z)*180/math.pi:.2f}°\nR = {z.real:.2f} Ω\nX = {z.imag:.2f} Ω"
                self._cursors.append(PlotCursor(self.canvas_zin, [ax1, ax2], f_z, [fmt_zin]))
            self.fig_zin.tight_layout()
            self.canvas_zin.draw()

            self.fig_gamma.clear()
            ax_g = self.fig_gamma.add_subplot(111)
            valid_g = [(f, g) for f, g in zip(freqs_mhz, sweep.gamma_mag, strict=True) if g is not None]
            if valid_g:
                f_g = [x[0] for x in valid_g]
                g_vals = [x[1] for x in valid_g]
                ax_g.plot(f_g, g_vals)
                ax_g.set_ylabel("|Γ|")
                def fmt_gamma(idx):
                    g = g_vals[idx]
                    rl = -20 * math.log10(g) if g > 1e-12 else float('inf')
                    return f"f = {f_g[idx]:.4f} MHz\n|Γ| = {g:.4f}\nRL = {rl:.2f} dB"
                self._cursors.append(PlotCursor(self.canvas_gamma, ax_g, f_g, [fmt_gamma]))
            ax_g.set_xlabel("Frequency (MHz)")
            ax_g.set_title("Reflection Coefficient")
            ax_g.grid(True, alpha=0.3)
            add_overlays(ax_g, is_gamma=True)
            self.fig_gamma.tight_layout()
            self.canvas_gamma.draw()

            self.fig_veom.clear()
            ax_v = self.fig_veom.add_subplot(111)
            valid_v = [(f, v) for f, v in zip(freqs_mhz, sweep.v_eom_mag, strict=True) if v is not None]
            if valid_v:
                f_v = [x[0] for x in valid_v]
                v_vals = [x[1] for x in valid_v]
                ax_v.plot(f_v, v_vals)
                ax_v.set_ylabel("|V_EOM| (V)")
                def fmt_veom(idx):
                    fv = f_v[idx]
                    v = v_vals[idx]
                    nearest = min(targets, key=lambda t: abs(t['freq_hz']/1e6 - fv)) if targets else None
                    if nearest and nearest['v_rms'] is not None:
                        tgt_v = nearest['v_rms']
                        err = (v - tgt_v) / tgt_v * 100 if tgt_v else 0
                        return f"f = {fv:.4f} MHz\nV_EOM = {v:.4f} V RMS\ntarget = {tgt_v:.4f} V RMS\nerror = {err:.2f} %"
                    return f"f = {fv:.4f} MHz\nV_EOM = {v:.4f} V RMS"
                self._cursors.append(PlotCursor(self.canvas_veom, ax_v, f_v, [fmt_veom]))
            ax_v.set_xlabel("Frequency (MHz)")
            ax_v.set_title("EOM Voltage vs Frequency")
            ax_v.grid(True, alpha=0.3)
            add_overlays(ax_v, is_voltage=True)
            self.fig_veom.tight_layout()
            self.canvas_veom.draw()
        except Exception:
            pass  # degrade gracefully if sweep format unexpected
