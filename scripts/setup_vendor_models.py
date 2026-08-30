#!/usr/bin/env python3
"""
Bootstrap script to generate the local Real Vendor Database.
This script expects the proprietary vendor packs to be either in vendor_packs/ or ~/Downloads.
It creates a production-ready SQLite library database.
"""
import shutil
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from foster_eom.catalog.library import ComponentLibrary  # noqa: E402
from foster_eom.catalog.vendor_pack import VendorPackSpec  # noqa: E402
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl  # noqa: E402

OFFICIAL_URLS = {
    "coilcraft_ads_rf_library.zip": "https://www.coilcraft.com/models",
    "gjm-s-v77.zip": "https://www.murata.com/en-us/tool/data/s-parameter",
    "gqm-s-v77.zip": "https://www.murata.com/en-us/tool/data/s-parameter"
}

ARCHIVES = [
    "coilcraft_ads_rf_library.zip",
    "gjm-s-v77.zip",
    "gqm-s-v77.zip"
]

def main():
    packs_dir = project_root / "vendor_packs"
    lib_dir = project_root / "vendor_libraries"
    db_path = lib_dir / "posm_vendor_components.fseom.db"
    downloads_dir = Path.home() / "Downloads"

    packs_dir.mkdir(exist_ok=True)
    lib_dir.mkdir(exist_ok=True)

    # 1. Locate / Copy archives
    print("--- Searching for Vendor Archives ---")
    all_found = True
    for archive in ARCHIVES:
        dest = packs_dir / archive
        if dest.exists():
            print(f"FOUND: {dest}")
            continue

        src = downloads_dir / archive
        if src.exists():
            print(f"FOUND in Downloads: {src} -> Copying to {dest}")
            shutil.copy2(src, dest)
            continue

        print(f"MISSING: {archive}")
        print(f"  -> Please download from: {OFFICIAL_URLS[archive]}")
        all_found = False

    if not all_found:
        print("\nERROR: Cannot proceed without all vendor packs.")
        sys.exit(1)

    # 2. Build Library
    print("\n--- Initializing Library ---")
    if db_path.exists():
        print(f"Removing old DB: {db_path}")
        db_path.unlink()

    lib = ComponentLibrary(str(db_path))
    lib.close()

    print(f"Created empty DB at {db_path}")

    # 3. Import
    specs = [
        VendorPackSpec(
            vendor="Coilcraft",
            adapter="s2p_coilcraft",
            source_path=packs_dir / "coilcraft_ads_rf_library.zip",
            glob_pattern="**/*.s2p",
            measurement_plane="EOM_external_RF_connector"
        ),
        VendorPackSpec(
            vendor="Murata",
            adapter="s2p_murata_gjm_gqm",
            source_path=packs_dir / "gjm-s-v77.zip",
            glob_pattern="**/*.s2p",
            measurement_plane="EOM_external_RF_connector"
        ),
        VendorPackSpec(
            vendor="Murata",
            adapter="s2p_murata_gjm_gqm",
            source_path=packs_dir / "gqm-s-v77.zip",
            glob_pattern="**/*.s2p",
            measurement_plane="EOM_external_RF_connector"
        )
    ]

    print("\n--- Importing Vendor Packs ---")
    for spec in specs:
        print(f"\nImporting {spec.vendor} from {spec.source_path.name}...")
        try:
            res = LibraryCtrl.import_pack(spec, str(db_path))
            print(f"  Files discovered: {res.n_files_processed}")
            print(f"  Inserted: {res.n_inserted_total}")
            print(f"  Duplicates skipped: {res.n_skipped_dup_total}")
            print(f"  Warnings: {0}")
            print(f"  Errors: {res.n_error_total}")
            if res.all_errors:
                print("  Sample errors:")
                for e in res.all_errors[:3]:
                    print(f"   - {e}")
        except Exception as e:
            print(f"  FATAL ERROR during import: {e}")

    # 4. Verification Stats
    print("\n--- Verification ---")
    stats = LibraryCtrl.get_stats(str(db_path))
    print(f"DB SHA256: {stats.sha256}")
    print(f"Total Parts: {stats.total_parts}")
    print(f"  Inductors: {stats.n_inductors}")
    print(f"  Capacitors: {stats.n_capacitors}")
    print(f"  Measured: {stats.n_measured}")
    print(f"Vendors: {set(p.vendor for p in stats.parts)}")

    print("\nSUCCESS: Real-vendor DB is ready.")
    print(f"-> You can now open {db_path.relative_to(project_root)} in the GUI.")

if __name__ == "__main__":
    main()
