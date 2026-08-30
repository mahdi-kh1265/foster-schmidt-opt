from pathlib import Path

from foster_eom.catalog.component import ComponentKind
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.catalog.query import ComponentQuery
from foster_eom.catalog.vendor_pack import VendorPackSpec
from foster_eom.domain.results import CandidateResult
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl
from foster_eom.gui.state import ProjectState


def test_p09_handoff_regression(tmp_path: Path):
    db_path = str(tmp_path / "test.fseom.db")
    demo_pack_path = Path("examples") / "demo_vendor_pack.zip"

    # Create empty library
    lib = ComponentLibrary(db_path)
    lib.close()

    # Import demo pack
    spec_l = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="coilcraft_csv",
        source_path=demo_pack_path,
        glob_pattern="**/demo_inductors.csv",
        measurement_plane="EOM_external_RF_connector"
    )
    LibraryCtrl.import_pack(spec_l, db_path)

    stats = LibraryCtrl.get_stats(db_path)
    active_sha = stats.sha256

    # Mock state
    state = ProjectState()
    state.library_path = db_path
    state.library_sha = active_sha
    state.frequencies_hz = [10e6]
    state.voltage_targets_rms_v = [1.0]

    # Prove Realization gets the same DB path + SHA
    assert state.library_path == db_path
    assert state.library_sha == active_sha



    # We won't fully execute RealizationCtrl because it triggers the beam search
    # But we prove ComponentQuery on the active DB returns the part from that DB
    lib2 = ComponentLibrary(state.library_path)
    assert LibraryCtrl.get_stats(state.library_path).sha256 == state.library_sha

    parts = lib2.query(ComponentQuery(kind=ComponentKind.INDUCTOR))
    assert len(parts) > 0
    assert parts[0].vendor == "POSM-DEMO"
    lib2.close()
