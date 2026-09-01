
from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.state import (
    EOMParams,
    MatchParams,
    OptimizationPresetParams,
    ProjectState,
    SourceParams,
    StressParams,
    TopologyParams,
)


def test_gui_acceptance_end_to_end_feasible():
    """P12-GUI2 Acceptance Fixture: End-to-End Real Pipeline Test.
    
    Constructs a fully GUI-enterable MVC ProjectState, converts it,
    and runs the unmocked optimization engine to prove feasibility
    and data-model provenance.
    """
    # 1. Construct exact GUI-compatible ProjectState
    state = ProjectState()
    state.name = "P12 GUI Acceptance Fixture"
    state.frequencies_hz = [10e6]
    state.voltage_targets_rms_v = [10.0]
    state.sweep_f_min_hz = 5e6
    state.sweep_f_max_hz = 15e6

    state.source = SourceParams(mode="thevenin", vth_rms=1.0, z_source_ohm=50.0)

    # Must use lossy_capacitor to introduce non-zero Rs.
    # If Rs=0, parallel equivalent resistance is 0, causing SCHMIDT_INFEASIBLE
    # because R_p must be >= R_match (50 Ohm) for the Schmidt Shunt-Then-Series
    # orientation to be structurally feasible.
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

    # FAST preset to keep test duration reasonable while still engaging DE + Polish
    state.optimization_preset = OptimizationPresetParams(
        preset="FAST",
        custom_max_global_evaluations=500,
        custom_polish_top_k=2,
        custom_local_max_iterations=50
    )

    state.input_sha256 = state.compute_input_sha()

    # 2. Run real optimization pipeline (covers state_to_spec and run_optimization)
    res = OptimizeCtrl.run(state)

    # 3. Assertions
    assert len(res.candidates) > 0, "Expected at least one candidate"

    best = res.candidates[0]

    # Ensure local polish ran and the provenance string is populated
    assert best.local_polish_method is not None
    assert best.local_polish_method != ""
    assert isinstance(best.local_polish_method, str)

    # Ensure constraint labels remain canonical (hard_N)
    margins = best.constraint_margins
    assert len(margins) > 0, "Expected constraint margins to be populated"
    for key in margins.keys():
        assert key.startswith("hard_"), f"Expected canonical constraint key, got: {key}"
