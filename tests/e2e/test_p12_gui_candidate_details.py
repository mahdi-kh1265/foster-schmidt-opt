import pytest

from foster_eom.gui.pages.synthesize_page import SynthesizePage
from foster_eom.gui.state import (
    EOMParams,
    ProjectState,
    SourceParams,
    TopologyParams,
    MatchParams,
    StressParams,
    OptimizationPresetParams,
)
from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.optimize.engine import OptimizationResult
from PySide6.QtWidgets import QApplication

def test_candidate_details_population():
    """Verify that selecting a candidate in the SynthesizePage table
    populates the detail text view without throwing an ImportError,
    and includes all required provenance strings.
    """
    # 1. Construct known-good state
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
    # ultra-fast preset just to get a candidate
    state.optimization_preset = OptimizationPresetParams(
        preset="FAST",
        custom_max_global_evaluations=10,
        custom_polish_top_k=1,
        custom_local_max_iterations=10
    )
    
    # 2. Run engine synchronously
    res: OptimizationResult = OptimizeCtrl.run(state)
    assert len(res.candidates) > 0
    
    app = QApplication.instance() or QApplication([])
    
    # 3. Create the GUI page and inject state/results
    page = SynthesizePage()
    page.set_state(state)
    page._on_finished(res)  # Mock the worker completion
    
    # Ensure the detail text is initially empty
    page.detail_text.clear()
    assert page.detail_text.toPlainText() == ""
    
    # 4. Trigger selection (Candidate #1 is row 0)
    page._on_selection(0)
    
    # 5. Assert detail text is populated
    text = page.detail_text.toPlainText()
    assert text != "", "Selected Candidate Details panel is blank."
    
    # 6. Check required items
    assert "Candidate #1" in text
    assert "Objective:" in text
    assert "Feasible:" in text
    assert "Numerical:" in text
    assert "Local polish:" in text
    assert "Hard constraints:" in text
