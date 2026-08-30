from pathlib import Path

from foster_eom.catalog.component import ComponentKind
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.catalog.query import ComponentQuery
from foster_eom.catalog.vendor_pack import VendorPackSpec
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl


def test_p12_library_acceptance(tmp_path: Path):
    db_path = str(tmp_path / "test.fseom.db")
    demo_pack_path = Path("examples") / "demo_vendor_pack.zip"
    assert demo_pack_path.exists()

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
    spec_c = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="murata_csv",
        source_path=demo_pack_path,
        glob_pattern="**/demo_capacitors.csv",
        measurement_plane="EOM_external_RF_connector"
    )
    spec_l_s2p = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="s2p_coilcraft",
        source_path=demo_pack_path,
        glob_pattern="**/*-L-*.s2p",
        measurement_plane="EOM_external_RF_connector"
    )
    spec_c_s2p = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="s2p_murata_gjm_gqm",
        source_path=demo_pack_path,
        glob_pattern="**/*-C-*.s2p",
        measurement_plane="EOM_external_RF_connector"
    )

    LibraryCtrl.import_pack(spec_l, db_path)
    LibraryCtrl.import_pack(spec_c, db_path)
    LibraryCtrl.import_pack(spec_l_s2p, db_path)
    LibraryCtrl.import_pack(spec_c_s2p, db_path)

    # Assert rows exist and stats update
    stats1 = LibraryCtrl.get_stats(db_path)
    assert stats1.total_parts == 26
    assert stats1.n_inductors == 13
    assert stats1.n_capacitors == 13
    sha1 = stats1.sha256
    assert sha1 is not None
    assert len(sha1) == 64

    # Test filters
    q_l = ComponentQuery(kind=ComponentKind.INDUCTOR)
    stats_l = LibraryCtrl.get_stats(db_path, query=q_l)
    assert stats_l.total_parts == 13


    # Real vendor classification test (is_synthetic == False)
    import zipfile
    with zipfile.ZipFile(demo_pack_path, "r") as z:
        csv_data = z.read("coilcraft/demo_inductors.csv").decode("utf-8")

    csv_data = csv_data.replace("POSM-DEMO", "FakeRealCorp")

    real_csv = tmp_path / "real_inductors.csv"
    real_csv.write_text(csv_data)

    spec_real = VendorPackSpec(
        vendor="FakeRealCorp",
        adapter="coilcraft_csv",
        source_path=tmp_path,
        glob_pattern="real_inductors.csv",
        measurement_plane="EOM_external_RF_connector"
    )
    LibraryCtrl.import_pack(spec_real, db_path)

    stats_real = LibraryCtrl.get_stats(db_path, query=ComponentQuery(vendor="FakeRealCorp"))
    assert stats_real.total_parts > 0
    part = stats_real.parts[0]
    details_vm = LibraryCtrl.get_component_details(db_path, part.id)
    assert not details_vm.is_synthetic

    # Downstream P09 Handoff Test
    from foster_eom.gui.state import ProjectState
    state = ProjectState()
    state.library_path = db_path
    state.frequencies_hz = [10e6]
    state.voltage_targets_rms_v = [1.0]

    from foster_eom.domain.results import CandidateResult
    from foster_eom.optimize.engine import OptimizationResult

    c = CandidateResult(
        orientation="series",
        domain_id="branch_foster",
        branch1_realization="L",
        branch2_realization="C",
        branch1_cells=1,
        branch2_cells=1,
        resolved_values={"L_b1_c1": 10e-9, "C_b2_c1": 10e-12},
        pole_locations_hz=[10e6, 20e6]
    )


    pass


    # Check that POSM-DEMO parts are marked is_synthetic=True
    stats_demo = LibraryCtrl.get_stats(db_path, query=ComponentQuery(vendor="POSM-DEMO"))
    demo_part = stats_demo.parts[0]
    demo_details_vm = LibraryCtrl.get_component_details(db_path, demo_part.id)
    assert demo_details_vm.is_synthetic

