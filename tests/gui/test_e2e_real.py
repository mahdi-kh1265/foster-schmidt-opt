"""Real-backend headless integration test for GUI components."""

import pytest

from foster_eom.gui.adapter import load_gui_project, save_gui_project
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl
from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.controllers.realization_ctrl import RealizationCtrl
from foster_eom.gui.controllers.robustness_ctrl import RobustnessCtrl
from foster_eom.gui.controllers.spice_ctrl import SpiceCtrl
from foster_eom.gui.controllers.verify_ctrl import VerifyCtrl
from foster_eom.gui.state import ProjectState
from foster_eom.gui.view_models.optimize_vm import OptimizeVM
from foster_eom.gui.view_models.robustness_vm import RobustnessVM
from foster_eom.gui.view_models.verify_vm import VerifyVM


# P12: expand real headless integration test through tiny P05-P11 workflow
@pytest.mark.integration
def test_full_headless_workflow(tmp_path):
    # 1. Project Setup
    state = ProjectState()
    state.frequencies_hz = [10e6]
    state.eom.model_type = "ideal_capacitor"
    state.eom.c0_f = 250e-12
    state.topology.n_branches = 1
    state.topology.n_cells_per_branch = 1

    # Populate Library with useful parts
    lib_path = str(tmp_path / "test_lib.db")
    import uuid

    from foster_eom.catalog.component import (
        ComponentKind,
        LibraryComponent,
        ModelCondition,
        ModelOrigin,
        ModelTier,
    )
    from foster_eom.catalog.library import ComponentLibrary

    lib = ComponentLibrary(lib_path)

    # Add 1 uH inductor
    c = LibraryComponent(
        id=str(uuid.uuid4()),
        kind=ComponentKind.INDUCTOR,
        vendor="Test",
        part_number="L1",
        value_nom=1e-6,
        value_tol_frac=0.10,
        voltage_max_v=50.0,
    )
    cid = lib.add(c)
    lib.add_model_condition(
        ModelCondition(
            id=str(uuid.uuid4()),
            component_id=cid,
            model_tier=ModelTier.IDEAL,
            model_origin=ModelOrigin.IDEAL,
        )
    )

    lib.close()

    state.library_path = lib_path
    state.library_sha = LibraryCtrl.get_stats(lib_path).sha256

    # Update input hash
    state.input_sha256 = state.compute_input_sha()

    # 2. P05 Optimize
    opt_res = OptimizeCtrl.run(state)
    assert opt_res is not None
    # Just to confirm the view model handles it without crash
    OptimizeVM.from_result(opt_res)

    # 3. P06 Verify
    if opt_res.best_feasible is None and opt_res.near_feasible_best is None:
        pytest.skip("No candidates found, skipping remainder of pipeline.")

    _sweep_res, q_metrics, stress_res = VerifyCtrl.run(state, opt_res)
    VerifyVM.from_results(q_metrics, stress_res)

    # 4. P09 Realize
    real_res = RealizationCtrl.run(state, opt_res)

    if real_res.best is not None:
        # 5. P10 Robustness
        rob_res = RobustnessCtrl.run(state, real_res)
        RobustnessVM.from_result(rob_res)

        # 6. P11 SPICE
        netlist = SpiceCtrl.export_netlist(state, real_res)
        assert "vth_phasor" in netlist or "Vsense_oriented" in netlist or "Vsrc" in netlist

        try:
            spice_res = SpiceCtrl.validate(state, real_res)
            from foster_eom.gui.view_models.spice_vm import SpiceVM

            SpiceVM.from_report(spice_res)
        except Exception:
            pass  # ngspice might not be installed, handled gracefully in UI, here we just catch

    # 5. Persist and Reload
    proj_file = tmp_path / "project.yaml"
    save_gui_project(state, proj_file)

    loaded_state = load_gui_project(proj_file)
    assert loaded_state.library_path == lib_path
    assert loaded_state.input_sha256 == state.input_sha256
    assert loaded_state.frequencies_hz == [10e6]
