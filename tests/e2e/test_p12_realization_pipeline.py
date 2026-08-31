import math
import sqlite3
import pytest
from pathlib import Path

from foster_eom.gui.state import ProjectState
from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
from foster_eom.gui.controllers.realization_ctrl import RealizationCtrl
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl
from foster_eom.catalog.vendor_pack import VendorPackSpec

def test_realization_pipeline_synthetic(tmp_path):
    """
    P12 REALIZATION - REGRESSION TEST
    
    Proves the production P09 RealizationCtrl path without mocking.
    Because the real Murata DB lacks coverage at 10 MHz (validity starts at 100 MHz)
    and max C is 150 pF, we use the synthetic demo catalog to prove pipeline success.
    
    This preserves the 10 MHz hardware target without falsifying real catalog data.
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
    
    # Custom fast preset to ensure predictable optimization result
    state.optimization_preset.preset = "CUSTOM"
    state.optimization_preset.custom_max_global_evaluations = 250
    state.optimization_preset.custom_local_max_iterations = 2
    
    db_path = str(tmp_path / "test_synthetic.fseom.db")
    lib = ComponentLibrary(db_path)
    lib.close()
    
    root = Path(__file__).parent.parent.parent
    
    spec_l = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="coilcraft_csv",
        source_path=root / "examples" / "demo_vendor_pack.zip",
        glob_pattern="**/demo_inductors.csv",
        measurement_plane="EOM_external_RF_connector"
    )
    spec_c = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="murata_csv",
        source_path=root / "examples" / "demo_vendor_pack.zip",
        glob_pattern="**/demo_capacitors.csv",
        measurement_plane="EOM_external_RF_connector"
    )
    
    LibraryCtrl.import_pack(spec_l, db_path)
    LibraryCtrl.import_pack(spec_c, db_path)
    
    # Patch demo DB to satisfy STRICT frequency validity at 10 MHz
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE model_conditions SET validity_hz_lo = 1000.0, validity_hz_hi = 10000000000.0")
    conn.commit()
    conn.close()
    
    stats = LibraryCtrl.get_stats(db_path)
    state.library_path = db_path
    state.library_sha = stats.sha256
    
    # Step 1: Continuous Optimization
    opt_res = OptimizeCtrl.run(state)
    assert len(opt_res.candidates) > 0
    
    # Step 2: Realization
    real_res = RealizationCtrl.run(state, opt_res)
    
    assert real_res.status == "feasible"
    assert real_res.best is not None
    assert real_res.diagnostics.total_combos > 0
    assert real_res.diagnostics.n_combos_evaluated > 0
    assert "b1_L1" in real_res.best.slot_entries
    assert "b1_C1" in real_res.best.slot_entries
    
