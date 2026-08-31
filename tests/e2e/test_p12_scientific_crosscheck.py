import math
import sqlite3
from pathlib import Path
import pytest

from foster_eom.gui.state import ProjectState
from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.controllers.realization_ctrl import RealizationCtrl
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl
from foster_eom.catalog.vendor_pack import VendorPackSpec

def test_real_vendor_rejection_regression(tmp_path):
    """
    P12 REALIZATION - REAL VENDOR NO-CANDIDATE REGRESSION

    Proves the 10 MHz benchmark correctly yields no_candidates against
    a catalog exhibiting real Murata bounds (C max 150pF, min freq 100MHz).
    """
    state = ProjectState()
    state.name = "P12 Synthetic Realization Test"
    state.frequencies_hz = [10e6]
    state.voltage_targets_rms_v = [10.0]
    state.sweep_f_min_hz = 5e6
    state.sweep_f_max_hz = 15e6
    state.source.mode = "thevenin"
    state.source.vth_rms = 1.0
    state.source.z_source_real_ohm = 50.0
    state.eom.model_type = "lossy_capacitor"
    state.eom.c0_f = 200e-12
    state.eom.rs_ohm = 10.0
    state.topology.n_branches = 2
    state.topology.n_cells_per_branch = 1

    state.optimization_preset.preset = "CUSTOM"
    state.optimization_preset.custom_max_global_evaluations = 250
    state.optimization_preset.custom_local_max_iterations = 2

    # We will build a dummy library that mirrors the limitations of the real one
    db_path = str(tmp_path / "test_reject.fseom.db")
    lib = ComponentLibrary(db_path)
    
    import uuid
    from datetime import datetime, UTC
    from foster_eom.catalog.component import ComponentKind, LibraryComponent, ModelCondition, ModelTier, ModelOrigin
    
    now = datetime.now(UTC).isoformat()
    # Add a mock capacitor simulating Murata GJM/GQM limitations: max 150pF, freq valid from 100MHz
    comp = LibraryComponent(
        id=str(uuid.uuid4()),
        kind=ComponentKind.CAPACITOR,
        vendor="Mock-Murata",
        part_number="MOCK-150pF",
        value_nom=150e-12, # Max 150pF
        value_tol_frac=0.05,
        voltage_max_v=50.0,
        current_max_a=None,
        current_sat_a=None,
        package="0402",
        description="",
        import_source="test",
        import_sha256="",
        import_ts=now
    )
    comp.content_sha256 = comp.compute_content_sha256()
    cid = lib.add(comp)
    
    mc = ModelCondition(
        id=str(uuid.uuid4()),
        component_id=cid,
        model_tier=ModelTier.MEASURED,
        model_origin=ModelOrigin.VENDOR_TOUCHSTONE,
        parametric_params={},
        srf_hz=None,
        q_at_f_hz=None,
        q_value=None,
        esr_ohm=None,
        validity_hz_lo=100e6, # Freq valid only starting at 100MHz
        validity_hz_hi=20e9,
        import_ts=now
    )
    lib.add_model_condition(mc)
    
    lib.close()

    stats = LibraryCtrl.get_stats(db_path)
    state.library_path = db_path
    state.library_sha = stats.sha256

    opt_res = OptimizeCtrl.run(state)
    assert len(opt_res.candidates) > 0

    real_res = RealizationCtrl.run(state, opt_res)
    
    assert real_res.status == "no_candidates"
    
    diag = real_res.diagnostics
    assert diag.total_combos == 0
    assert diag.n_combos_evaluated == 0
    
    assert "b1_C1" in diag.rejection_reasons
    assert "b2_C1" in diag.rejection_reasons
    assert "0 eligible" in diag.rejection_reasons["b1_C1"]
    
