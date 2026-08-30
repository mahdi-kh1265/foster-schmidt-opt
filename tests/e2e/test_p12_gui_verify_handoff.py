from PySide6.QtWidgets import QApplication

from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.controllers.verify_ctrl import VerifyCtrl
from foster_eom.gui.main_window import MainWindow
from foster_eom.gui.state import (
    EOMParams,
    MatchParams,
    OptimizationPresetParams,
    ProjectState,
    SourceParams,
    StressParams,
    TopologyParams,
)
from foster_eom.optimize.engine import OptimizationResult


def test_verify_handoff_live_session():
    """Verify live handoff from SynthesizePage to VerifyPage without saving/loading."""
    state = ProjectState()
    state.frequencies_hz = [10e6]
    state.voltage_targets_rms_v = [10.0]
    state.sweep_f_min_hz = 5e6
    state.sweep_f_max_hz = 15e6
    state.source = SourceParams(mode="thevenin", vth_rms=1.0, z_source_ohm=50.0)
    state.eom = EOMParams(model_type="lossy_capacitor", c0_f=200e-12, rs_ohm=10.0)
    state.topology = TopologyParams(n_branches=2, n_cells_per_branch=1)
    state.match_params = MatchParams(
        gamma_max=0.5,
        resistance_min_ohm=0.1,
        resistance_max_ohm=10000.0,
        max_abs_reactance_ohm=10000.0
    )
    state.stress_params = StressParams(
        source_current_rms_max_a=100.0,
        off_target_eom_peak_rms_v=100.0,
        default_cap_peak_voltage_v=1000.0,
        default_ind_peak_current_a=100.0
    )
    state.optimization_preset = OptimizationPresetParams(
        preset="FAST",
        custom_max_global_evaluations=500,
        custom_polish_top_k=2,
        custom_local_max_iterations=50
    )
    state.input_sha256 = state.compute_input_sha()

    _ = QApplication.instance() or QApplication([])

    window = MainWindow()
    window._state = state
    window.project_page.populate_from_state(state)
    window._update_downstream()

    assert not window.verify_page.btn_run.isEnabled()

    from typing import cast
    res = cast(OptimizationResult, OptimizeCtrl.run(state))
    assert len(res.candidates) > 0

    window.synthesize_page.set_state(state)
    window.synthesize_page._on_finished(res)

    assert window.verify_page.btn_run.isEnabled()
    assert window.verify_page._opt_result is res

    verify_res = VerifyCtrl.run(state, res)
    sweep_res, q_metrics, stress_res, z_in_sweep = verify_res

    assert sweep_res is not None
    # Now simulate the result coming back to VerifyPage
    window.verify_page._on_finished(verify_res)

    # 4. Assert UI populated tables
    assert window.verify_page.q_table.rowCount() >= 1
    # Check that canonical metrics are plotted (no dummy columns)
    assert window.verify_page.q_table.columnCount() == 5
    assert window.verify_page.stress_table.rowCount() >= 1

    # Assert plots are drawn
    assert len(window.verify_page.fig_zin.axes) == 2
    ax1, ax2 = window.verify_page.fig_zin.axes
    assert len(ax1.lines) > 0
    assert len(ax1.lines[0].get_xdata()) > 0
    assert len(ax2.lines) > 0
    assert len(ax2.lines[0].get_ydata()) > 0

    assert len(window.verify_page.fig_gamma.axes) == 1
    ax_g = window.verify_page.fig_gamma.axes[0]
    assert len(ax_g.lines) > 0

    assert len(window.verify_page.fig_veom.axes) == 1
    ax_v = window.verify_page.fig_veom.axes[0]
    assert len(ax_v.lines) > 0

    # 5. UI failure-state robustness regression
    window.verify_page.btn_run.setEnabled(False)
    window.verify_page.progress.setVisible(True)
    window.verify_page.lbl_status.setText("Running...")

    # Simulate a corrupted result returning from controller
    window.verify_page._on_finished(("bad_sweep", "bad_q", "bad_stress"))

    # The UI should have gracefully recovered its enabled state and shown an error
    assert window.verify_page.btn_run.isEnabled()
    assert not window.verify_page.progress.isVisible()
    assert window.verify_page.lbl_status.text() == "FAILED"
    assert "Rendering error" in window.verify_page.warn_label.text()
