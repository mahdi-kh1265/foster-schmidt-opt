import numpy as np

from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.controllers.verify_ctrl import VerifyCtrl
from foster_eom.gui.state import ProjectState


def test_p12_scientific_crosscheck_voltage_objective():
    """Prove that GUI V_EOM target survives into objective, and an unattainable target creates nonzero j_voltage."""
    state = ProjectState()
    state.frequencies_hz = [10e6]
    state.voltage_targets_rms_v = [10.0]  # Unattainable target
    state.sweep_f_min_hz = 5e6
    state.sweep_f_max_hz = 15e6

    state.source.vth_rms = 1.0
    state.source.z_source_ohm = 50.0

    state.eom.model_type = "lossy_capacitor"
    state.eom.c0_f = 200e-12
    state.eom.rs_ohm = 10.0

    state.topology.n_branches = 2
    state.topology.n_cells_per_branch = 1

    state.optimization_preset.preset = "CUSTOM"
    state.optimization_preset.custom_max_global_evaluations = 100
    state.optimization_preset.custom_local_max_iterations = 1
    state.objective_weights.weight_voltage = 1.0

    # Run optimization
    opt_res = OptimizeCtrl.run(state)
    assert opt_res is not None

    # Extract the top candidate
    cand = opt_res.candidates[0]

    # 1. Ensure target survived and generated nonzero j_voltage
    j_voltage = cand.objective_terms.get("j_voltage", 0.0)
    assert j_voltage > 0.1, f"Expected nonzero j_voltage for unattainable 10V target, got {j_voltage}"

    # Run verify
    verify_res = VerifyCtrl.run(state, opt_res)
    sweep_res, _q_metrics, stress_res, _z_in_sweep = verify_res

    # 2. Optimization V_EOM matches P06 V_EOM
    v_opt = cand.target_solution_summaries[0].v_eom_mag

    idx_10mhz = np.argmin(np.abs(np.array(sweep_res.frequencies_hz) - 10e6))
    v_p06 = sweep_res.v_eom_mag[idx_10mhz]

    # Note: P06 computes V_EOM peak or mag? v_eom_mag is magnitude (peak for 1V source)
    # Wait, the objective expects V_EOM RMS.
    # v_eom_mag is absolute value of the phasor. Since Vth_rms is 1.0, the phasor represents RMS.
    assert abs(v_opt - v_p06) < 1e-3, f"V_EOM mismatch: opt={v_opt}, p06={v_p06}"

    # 3. Sweep band obeys 5-15 MHz exactly
    assert min(sweep_res.frequencies_hz) >= 5e6
    assert max(sweep_res.frequencies_hz) <= 15e6

    # 4. Stress extrema are within 5-15 MHz
    for e in stress_res.elements:
        assert 5e6 <= e.sweep_worst_v_freq_hz <= 15e6

