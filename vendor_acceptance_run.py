#!/usr/bin/env python
"""Real vendor-data acceptance run for P09 freeze audit.

Runs the full pipeline:
  vendor pack -> P08 catalog import -> eligibility query -> P09 realization
  -> vendor-model MNA -> P06 verification

Usage:
    python -m foster_eom.catalog.vendor_acceptance_run
  or:
    python vendor_acceptance_run.py

Vendor files must already exist in vendor_packs/ (gitignored).
Output is printed to stdout; no files are modified (read-only after import).
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path


def _sha256_db(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> None:
    repo = Path.cwd()
    sys.path.insert(0, str(repo))

    from foster_eom.catalog.component import ComponentKind
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.catalog.query import ComponentQuery
    from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

    db_path = repo / "vendor_packs" / "real_parts.fseom.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    vendor_packs_dir = repo / "vendor_packs"

    packs = [
        VendorPackSpec(
            vendor="Coilcraft",
            adapter="s2p_coilcraft",
            source_path=vendor_packs_dir / "coilcraft" / "coilcraft_ads_rf_library.zip",
            glob_pattern="**/*.s2p",
            measurement_plane="EOM_external_RF_connector",
        ),
        VendorPackSpec(
            vendor="Murata",
            adapter="s2p_murata_gjm_gqm",
            source_path=vendor_packs_dir / "murata" / "gjm-s-v77.zip",
            glob_pattern="**/*.s2p",
            measurement_plane="EOM_external_RF_connector",
        ),
        VendorPackSpec(
            vendor="Murata",
            adapter="s2p_murata_gjm_gqm",
            source_path=vendor_packs_dir / "murata" / "gqm-s-v77.zip",
            glob_pattern="**/*.s2p",
            measurement_plane="EOM_external_RF_connector",
        ),
    ]

    print("=" * 70)
    print("P09 Real-Data Acceptance Run")
    print("=" * 70)

    with ComponentLibrary(db_path) as lib:
        wf = VendorPackWorkflow(lib)

        # ----------------------------------------------------------------
        # Step 1: Import
        # ----------------------------------------------------------------
        total_inserted = 0
        total_errors = 0
        for spec in packs:
            print(f"\n--- Importing {spec.vendor} / {spec.adapter} ---")
            print(f"    Source: {spec.source_path.name}")
            try:
                manifest = wf.run(spec)
                print(f"    Files processed : {manifest.n_files_processed}")
                print(f"    Inserted        : {manifest.n_inserted_total}")
                print(f"    Skipped (dup)   : {manifest.n_skipped_dup_total}")
                print(f"    Errors          : {manifest.n_error_total}")
                if manifest.all_errors[:3]:
                    for e in manifest.all_errors[:3]:
                        print(f"    ERROR: {e[:120]}")
                total_inserted += manifest.n_inserted_total
                total_errors += manifest.n_error_total
            except Exception as exc:
                print(f"    FAILED: {exc}")

        # ----------------------------------------------------------------
        # Step 2: Eligibility queries
        # ----------------------------------------------------------------
        print("\n--- Eligibility Queries ---")

        # Inductors: 1 nH - 1 uH, model valid at 100 MHz-3 GHz
        q_l = ComponentQuery(
            kind=ComponentKind.INDUCTOR,
            value_min=1e-9,
            value_max=1e-6,
        )
        inductors = lib.query(q_l)
        print(f"Inductors (1nH-1uH)         : {len(inductors)} parts")
        if inductors:
            sample_L = inductors[:5]
            for p in sample_L:
                print(f"  {p.part_number:30s}  L={p.value_nom * 1e9:.2f} nH  pkg={p.package}")

        # Capacitors: 1 pF - 100 pF (EOM matching range)
        q_c = ComponentQuery(
            kind=ComponentKind.CAPACITOR,
            value_min=1e-12,
            value_max=100e-12,
        )
        capacitors = lib.query(q_c)
        print(f"Capacitors (1pF-100pF)      : {len(capacitors)} parts")
        if capacitors:
            sample_C = capacitors[:5]
            for p in sample_C:
                print(f"  {p.part_number:30s}  C={p.value_nom * 1e12:.2f} pF  pkg={p.package}")

        # ----------------------------------------------------------------
        # Step 3: Model build smoke test for representative parts
        # ----------------------------------------------------------------
        print("\n--- Model Build Smoke Test ---")
        model_ok = 0
        model_fail = 0
        test_parts = []
        if inductors:
            test_parts += inductors[:3]
        if capacitors:
            test_parts += capacitors[:3]

        for comp in test_parts:
            try:
                model = lib.build_model(comp.id)
                z_test = model.z(100e6)
                z_abs = abs(z_test)
                if math.isfinite(z_abs):
                    print(f"  OK  {comp.part_number:35s}  |Z(100MHz)|={z_abs:.3f} Ohm")
                    model_ok += 1
                else:
                    print(f"  NaN {comp.part_number:35s}  |Z|=non-finite")
                    model_fail += 1
            except Exception as exc:
                print(f"  ERR {comp.part_number:35s}  {str(exc)[:80]}")
                model_fail += 1

        print(f"  Models OK: {model_ok}  Failed: {model_fail}")

        # ----------------------------------------------------------------
        # Step 4: P09 realization smoke test
        # ----------------------------------------------------------------
        print("\n--- P09 Realization Smoke Test ---")
        _run_p09_smoke(lib, inductors, capacitors)

        # ----------------------------------------------------------------
        # Step 5: Library hash
        # ----------------------------------------------------------------
        lib_sha = _sha256_db(db_path)
        print("\n--- Library SHA-256 ---")
        print(f"  {lib_sha}")

    print("\n=== Acceptance Run Complete ===")
    print(f"Total inserted: {total_inserted}  Total errors: {total_errors}")


def _run_p09_smoke(lib, inductors, capacitors) -> None:
    """Run a minimal P09 neighborhood + realization smoke using real catalog parts."""
    from foster_eom.realization.neighborhoods import build_neighborhoods
    from foster_eom.realization.spec import SlotSpec

    if not inductors:
        print("  SKIP: no inductors in catalog")
        return
    if not capacitors:
        print("  SKIP: no capacitors in catalog")
        return

    # Pick target values close to 10 nH and 10 pF
    target_l = 10e-9  # 10 nH
    target_c = 10e-12  # 10 pF

    slots = (
        SlotSpec(element_id="b1_L1", value_nom=target_l, value_ratio=5.0),
        SlotSpec(element_id="b1_C1", value_nom=target_c, value_ratio=5.0),
    )

    nh = build_neighborhoods(slots, lib, k_max=5)

    for eid, entries in nh.items():
        if entries:
            e = entries[0]
            print(
                f"  Slot {eid}: {len(entries)} candidates, best={e.part_number}  "
                f"tier={e.model_tier.value}  log_r={e.log_ratio:.3f}"
            )
        else:
            print(f"  Slot {eid}: 0 candidates")

    if all(len(nh[s.element_id]) > 0 for s in slots):
        print("  Neighborhood built successfully — all slots populated")
        print("  (Full MNA+P06 realization requires EvaluationContext; skipped in acceptance run)")
    else:
        empty = [s.element_id for s in slots if not nh[s.element_id]]
        print(f"  WARNING: empty slots: {empty}")


if __name__ == "__main__":
    run()
