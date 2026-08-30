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

    q_v = ComponentQuery(value_min=4e-9, value_max=20e-9)
    stats_v = LibraryCtrl.get_stats(db_path, query=q_v)
    assert stats_v.total_parts == 4

    # Component details
    c_id = stats1.parts[0].id
    details = LibraryCtrl.get_component_details(db_path, c_id)
    assert details.vendor == "POSM-DEMO"
    assert details.is_synthetic is True

    # Re-import dedup test
    LibraryCtrl.import_pack(spec_l, db_path)
    stats2 = LibraryCtrl.get_stats(db_path)
    assert stats2.total_parts == 26
    assert stats2.sha256 == sha1

    # Malformed pack test
    bad_pack = tmp_path / "bad.zip"
    bad_pack.write_bytes(b"PK\x05\x06" + b"\x00"*18)  # Empty ZIP
    spec_bad = VendorPackSpec(
        vendor="POSM-DEMO",
        adapter="coilcraft_csv",
        source_path=bad_pack,
        glob_pattern="**/*.csv",
    )
    manifest_bad = LibraryCtrl.import_pack(spec_bad, db_path)
    assert manifest_bad.n_inserted_total == 0

    # Query for Realization
    q_realize = ComponentQuery(
        kind=ComponentKind.INDUCTOR,
        value_min=1e-9,
        value_max=100e-9,
    )
    lib2 = ComponentLibrary(db_path)
    res = lib2.query(q_realize)
    lib2.close()
    assert len(res) >= 7

