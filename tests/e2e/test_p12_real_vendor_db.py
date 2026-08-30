from pathlib import Path

import pytest

from foster_eom.catalog.component import ComponentKind
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.catalog.query import ComponentQuery
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl


def test_p12_real_vendor_db_smoke():
    # Only run if the local DB was bootstrapped
    db_path = Path("vendor_libraries/posm_vendor_components.fseom.db")
    if not db_path.exists():
        pytest.skip("Local vendor DB not bootstrapped.")

    stats = LibraryCtrl.get_stats(str(db_path))
    assert stats.total_parts > 0
    assert "Coilcraft" in [p.vendor for p in stats.parts]
    assert "Murata" in [p.vendor for p in stats.parts]

    lib = ComponentLibrary(str(db_path))

    # query around 10 MHz
    q = ComponentQuery(kind=ComponentKind.INDUCTOR, freq_range_hz=(9e6, 11e6))
    parts = lib.query(q)

    assert len(parts) > 0, "Expected at least one inductor valid around 10 MHz"
    assert parts[0].vendor in ["Coilcraft", "Murata"]

    lib.close()
