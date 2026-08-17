"""Tests for vendor-pack import workflow (Prompt 09 / Freeze audit).

Tests use fully synthetic CSV files — no real manufacturer data required.
Real-data acceptance run is documented as PENDING EXTERNAL FILES.

Coverage
--------
TestVendorPackSpec      — VendorPackSpec defaults, adapter version
TestVendorPackWorkflow  — ZIP import, directory import, provenance fields,
                          manifest JSON round-trip, error handling,
                          idempotent re-import, source SHA stability
TestManifestJSON        — JSON serialization / deserialization
TestP09Invariants       — six frozen P09 contracts verified against actual code
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_coilcraft_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal Coilcraft-format CSV to *path*."""
    headers = [
        "Part Number",
        "Inductance",
        "Tolerance",
        "Irms",
        "Isat",
        "DCR Typ",
        "SRF Min",
        "Size",
        "Series",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def _write_murata_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal Murata-format CSV to *path*."""
    headers = [
        "Part Number",
        "Capacitance",
        "Tolerance",
        "Rated Voltage",
        "ESR",
        "Size",
        "Temperature Characteristic",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


_COILCRAFT_ROWS = [
    {
        "Part Number": "XAL5030-102MEH",
        "Inductance": "1.0e-6",
        "Tolerance": "0.20",
        "Irms": "4.0",
        "Isat": "6.0",
        "DCR Typ": "0.025",
        "SRF Min": "250e6",
        "Size": "XAL5030",
        "Series": "XAL",
    },
    {
        "Part Number": "XAL5030-222MEH",
        "Inductance": "2.2e-6",
        "Tolerance": "0.20",
        "Irms": "3.0",
        "Isat": "5.0",
        "DCR Typ": "0.040",
        "SRF Min": "150e6",
        "Size": "XAL5030",
        "Series": "XAL",
    },
]

_MURATA_ROWS = [
    {
        "Part Number": "GRM155R71H104KA88",
        "Capacitance": "100e-12",
        "Tolerance": "0.10",
        "Rated Voltage": "50",
        "ESR": "0.2",
        "Size": "0402",
        "Temperature Characteristic": "X7R",
    },
    {
        "Part Number": "GRM155R71H224KA12",
        "Capacitance": "220e-12",
        "Tolerance": "0.10",
        "Rated Voltage": "50",
        "ESR": "0.3",
        "Size": "0402",
        "Temperature Characteristic": "X7R",
    },
]


def _make_lib(tmp_path: Path):
    from foster_eom.catalog.library import ComponentLibrary

    return ComponentLibrary(tmp_path / "test.fseom.db")


# ---------------------------------------------------------------------------
# TestVendorPackSpec
# ---------------------------------------------------------------------------


class TestVendorPackSpec:
    def test_defaults_csv_glob(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec

        csv_path = tmp_path / "fake.csv"
        csv_path.touch()
        spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=csv_path)
        assert spec.glob_pattern == "**/*.csv"

    def test_defaults_s1p_glob(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec

        p = tmp_path / "fake.s1p"
        p.touch()
        spec = VendorPackSpec(vendor="V", adapter="touchstone", source_path=p)
        assert spec.glob_pattern == "**/*.s1p"

    def test_adapter_version_populated(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec

        p = tmp_path / "x"
        p.touch()
        spec = VendorPackSpec(vendor="V", adapter="coilcraft_csv", source_path=p)
        assert spec.adapter_version == "1.0.0"

    def test_unknown_adapter_version_is_unknown(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec

        p = tmp_path / "x"
        p.touch()
        spec = VendorPackSpec(vendor="V", adapter="custom_xyz", source_path=p)
        assert spec.adapter_version == "unknown"

    def test_path_coercion(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec

        spec = VendorPackSpec(
            vendor="V",
            adapter="coilcraft_csv",
            source_path=str(tmp_path),  # type: ignore[arg-type]
        )
        assert isinstance(spec.source_path, Path)


# ---------------------------------------------------------------------------
# TestVendorPackWorkflow (synthetic CSV, no real vendor files)
# ---------------------------------------------------------------------------


class TestVendorPackWorkflow:
    def _make_coilcraft_dir(self, base: Path) -> Path:
        """Create a directory containing a Coilcraft CSV."""
        d = base / "coilcraft_pack"
        d.mkdir()
        _write_coilcraft_csv(d / "XAL_series.csv", _COILCRAFT_ROWS)
        return d

    def _make_coilcraft_zip(self, base: Path) -> Path:
        """Create a ZIP containing a Coilcraft CSV."""
        d = self._make_coilcraft_dir(base)
        zip_path = base / "coilcraft_pack.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(d / "XAL_series.csv", arcname="XAL_series.csv")
        return zip_path

    def test_directory_import_inserts_parts(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        assert manifest.n_inserted_total >= 2
        assert manifest.n_files_processed == 1

    def test_zip_import_inserts_parts(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        zp = self._make_coilcraft_zip(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=zp)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        assert manifest.n_inserted_total >= 2

    def test_provenance_fields_populated(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(
                vendor="Coilcraft",
                adapter="coilcraft_csv",
                source_path=src,
                measurement_plane="EOM_external_RF_connector",
                extra_meta={"project": "foster-eom", "dut": "EOM-5000"},
            )
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        # Core provenance
        assert manifest.vendor == "Coilcraft"
        assert manifest.adapter == "coilcraft_csv"
        assert manifest.adapter_version == "1.0.0"
        assert manifest.measurement_plane == "EOM_external_RF_connector"
        assert len(manifest.source_sha256) == 64  # hex SHA-256
        assert manifest.import_timestamp_utc  # non-empty
        assert manifest.extra_meta["project"] == "foster-eom"
        assert len(manifest.library_sha256) == 64

    def test_per_file_sha_recorded(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        assert len(manifest.file_records) == 1
        rec = manifest.file_records[0]
        assert len(rec.sha256) == 64
        assert rec.filename.endswith(".csv")

    def test_idempotent_reimport(self, tmp_path: Path) -> None:
        """Same SHA → idempotent no-op on second import."""
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            m1 = wf.run(spec)
            m2 = wf.run(spec)

        assert m1.n_inserted_total >= 2
        # Second run: all duplicates
        assert m2.n_inserted_total == 0
        assert m2.n_skipped_dup_total >= 2

    def test_manifest_written_to_disk(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        manifest_dir = lib.db_path.parent / "import_manifests"
        files = list(manifest_dir.glob("*.json"))
        assert len(files) == 1
        loaded = json.loads(files[0].read_text())
        assert loaded["vendor"] == "Coilcraft"
        assert loaded["run_id"] == manifest.run_id

    def test_murata_csv_import(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        d = tmp_path / "murata_pack"
        d.mkdir()
        _write_murata_csv(d / "GRM_series.csv", _MURATA_ROWS)

        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Murata", adapter="murata_csv", source_path=d)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        assert manifest.n_inserted_total >= 2

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        bad_path = tmp_path / "does_not_exist.zip"
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="V", adapter="coilcraft_csv", source_path=bad_path)
            wf = VendorPackWorkflow(lib)
            with pytest.raises(FileNotFoundError, match="not found"):
                wf.run(spec)

    def test_unknown_adapter_raises(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import _make_importer

        with pytest.raises(ValueError, match="Unknown adapter"):
            _make_importer("nonexistent_adapter")

    def test_source_sha_stable_for_same_file(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import _sha256_path

        f = tmp_path / "test.csv"
        f.write_bytes(b"header\nrow1\nrow2\n")
        sha1 = _sha256_path(f)
        sha2 = _sha256_path(f)
        assert sha1 == sha2
        assert len(sha1) == 64

    def test_source_sha_differs_for_modified_file(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import _sha256_path

        f = tmp_path / "test.csv"
        f.write_bytes(b"original content")
        sha1 = _sha256_path(f)
        f.write_bytes(b"modified content")
        sha2 = _sha256_path(f)
        assert sha1 != sha2

    def test_summary_string_contains_vendor(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        summary = manifest.summary()
        assert "Coilcraft" in summary
        assert "coilcraft_csv" in summary
        assert "Files processed" in summary

    def test_parts_queryable_after_import(self, tmp_path: Path) -> None:
        """Imported Coilcraft inductors must be retrievable via eligibility query."""
        from foster_eom.catalog.component import ComponentKind
        from foster_eom.catalog.query import ComponentQuery
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            wf.run(spec)

            q = ComponentQuery(kind=ComponentKind.INDUCTOR, value_min=0.5e-6, value_max=5e-6)
            results = lib.query(q)

        assert len(results) >= 2
        pns = {r.part_number for r in results}
        assert "XAL5030-102MEH" in pns or any("XAL" in pn for pn in pns)

    def test_parts_reach_p09_neighborhoods(self, tmp_path: Path) -> None:
        """Coilcraft L parts imported via vendor pack appear in P09 slot neighborhoods."""
        from foster_eom.catalog.vendor_pack import VendorPackSpec, VendorPackWorkflow
        from foster_eom.realization.neighborhoods import build_neighborhoods
        from foster_eom.realization.spec import SlotSpec

        src = self._make_coilcraft_dir(tmp_path)
        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            VendorPackWorkflow(lib).run(spec)

            slot = SlotSpec(element_id="b1_L1", value_nom=1e-6, value_ratio=3.0)
            nh = build_neighborhoods((slot,), lib, k_max=5)

        assert len(nh["b1_L1"]) >= 1
        # Provenance binding verified
        for entry in nh["b1_L1"]:
            assert entry.component_id
            assert entry.model_condition_id
            assert entry.vendor == "Coilcraft"


# ---------------------------------------------------------------------------
# TestManifestJSON
# ---------------------------------------------------------------------------


class TestManifestJSON:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        from foster_eom.catalog.vendor_pack import (
            VendorPackManifest,
            VendorPackSpec,
            VendorPackWorkflow,
        )

        src = tmp_path / "coilcraft"
        src.mkdir()
        _write_coilcraft_csv(src / "data.csv", _COILCRAFT_ROWS)

        with _make_lib(tmp_path) as lib:
            spec = VendorPackSpec(vendor="Coilcraft", adapter="coilcraft_csv", source_path=src)
            wf = VendorPackWorkflow(lib)
            manifest = wf.run(spec)

        text = manifest.to_json()
        loaded = VendorPackManifest.from_json(text)
        assert loaded.vendor == manifest.vendor
        assert loaded.source_sha256 == manifest.source_sha256
        assert loaded.n_inserted_total == manifest.n_inserted_total
        assert loaded.library_sha256 == manifest.library_sha256
        assert len(loaded.file_records) == len(manifest.file_records)


# ---------------------------------------------------------------------------
# TestP09Invariants — frozen contract verification against actual code
# ---------------------------------------------------------------------------


class TestP09Invariants:
    """Verify the six P09 frozen invariants against actual implementation."""

    def test_invariant_1_no_movable_fp_reoptimization(self) -> None:
        """No reoptimize.py or allow_movable_pole_reopt in realization package."""
        from pathlib import Path

        realization_dir = Path(__file__).parent.parent.parent / "foster_eom" / "realization"
        for py_file in realization_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            assert "reoptimize" not in source, (
                f"{py_file.name} contains 'reoptimize' — stale P09 reference"
            )
            assert "allow_movable_pole" not in source, (
                f"{py_file.name} contains 'allow_movable_pole' — stale reference"
            )
            assert "movable_pole" not in source, (
                f"{py_file.name} contains 'movable_pole' — stale reference"
            )

    def test_invariant_2_exact_p05_deb_ranking(self) -> None:
        """runner.py calls dedup.deb_key (the exact P05 function) for ranking."""
        from pathlib import Path

        runner_src = (
            Path(__file__).parent.parent.parent / "foster_eom" / "realization" / "runner.py"
        ).read_text(encoding="utf-8")

        # Must import deb_key from optimize.dedup — the exact P05 module
        assert "from foster_eom.optimize.dedup import deb_key" in runner_src
        # Must call deb_key on evaluation result (not some local reimplementation)
        assert "deb_key(eval_result)" in runner_src
        # sort must use .deb_key attribute (stored on CatalogCombo)
        assert "key=lambda cc: cc.deb_key" in runner_src

    def test_invariant_2_deb_key_tuple_shape(self) -> None:
        """deb_key returns (not feasible, v_max, v_sum, obj) — exact P05 ordering."""
        from foster_eom.optimize.dedup import deb_key
        from foster_eom.optimize.evaluator import EvaluationResult

        r = EvaluationResult(
            x=(),
            objective_value=0.5,
            base_objective_value=0.5,
            soft_penalty_total=0.0,
            objective_terms={},
            hard_margins=(),
            soft_penalties={},
            v_max=0.0,
            v_sum=0.0,
            feasible=True,
            near_feasible=True,
            numerical_status="ok",
            numerical_failure_reason=None,
            failed_frequency_hz=None,
            failed_stage=None,
            all_solutions=(),
            target_solutions=(),
            coarse_evaluated=False,
        )
        k = deb_key(r)
        assert k == (False, 0.0, 0.0, 0.5)

    def test_invariant_3_exhaustive_vs_truncated_semantics(self) -> None:
        """generate_combos sets exhaustive/truncated flags correctly; infeasible
        status is never emitted from a truncated beam search."""
        from foster_eom.catalog.component import ModelTier
        from foster_eom.realization.beam import generate_combos
        from foster_eom.realization.spec import NeighborhoodEntry, RealizationSpec

        def _entry(i: int) -> NeighborhoodEntry:
            return NeighborhoodEntry(
                component_id=f"c{i}",
                model_condition_id=f"mc{i}",
                vendor="V",
                part_number=f"P{i}",
                value_nom=1e-9,
                value_tol_frac=0.05,
                model_tier=ModelTier.IDEAL,
                log_ratio=0.0,
            )

        # Case 1: product = 4 ≤ threshold 10 → exhaustive
        nh = {"b1_C1": [_entry(0), _entry(1)], "b1_L1": [_entry(2), _entry(3)]}
        spec = RealizationSpec(slot_specs=(), exhaustive_threshold=10)
        _, exhaustive, truncated = generate_combos(nh, spec)
        assert exhaustive is True
        assert truncated is False

        # Case 2: product = 9 > threshold 4 → beam/truncated
        nh3 = {"A": [_entry(i) for i in range(3)], "B": [_entry(i) for i in range(3)]}
        spec2 = RealizationSpec(slot_specs=(), exhaustive_threshold=4, beam_width=3)
        _, exhaustive2, truncated2 = generate_combos(nh3, spec2)
        assert exhaustive2 is False
        assert truncated2 is True

        # Verify that runner.py does NOT emit 'infeasible' from truncated search
        runner_src = (
            Path(__file__).parent.parent.parent / "foster_eom" / "realization" / "runner.py"
        ).read_text(encoding="utf-8")
        # The only 'infeasible' assignments must be guarded by search_exhaustive
        # (check that "infeasible" is always paired with search_exhaustive)
        assert 'status = "infeasible"' in runner_src
        # Verify the guard: infeasible must only appear inside search_exhaustive branches
        lines = runner_src.splitlines()
        for i, line in enumerate(lines):
            if '"infeasible"' in line and "status" in line:
                # Surrounding context must mention search_exhaustive
                context = " ".join(lines[max(0, i - 5) : i + 2])
                assert "search_exhaustive" in context, (
                    f"'infeasible' status set without search_exhaustive guard at line {i + 1}"
                )

    def test_invariant_4_component_and_mc_id_frozen_at_query_time(self) -> None:
        """NeighborhoodEntry is frozen=True dataclass with both component_id
        and model_condition_id fields populated at build_neighborhoods time."""
        import dataclasses

        from foster_eom.realization.spec import NeighborhoodEntry

        # Verify frozen=True
        assert NeighborhoodEntry.__dataclass_params__.frozen is True  # type: ignore[attr-defined]

        # Verify both ID fields exist
        field_names = {f.name for f in dataclasses.fields(NeighborhoodEntry)}
        assert "component_id" in field_names
        assert "model_condition_id" in field_names

        # Verify immutability
        from foster_eom.catalog.component import ModelTier

        entry = NeighborhoodEntry(
            component_id="cid1",
            model_condition_id="mc1",
            vendor="V",
            part_number="P1",
            value_nom=1e-9,
            value_tol_frac=0.05,
            model_tier=ModelTier.IDEAL,
            log_ratio=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.component_id = "mutated"  # type: ignore[misc]

    def test_invariant_5_p06_band_from_evaluation_frequencies(self) -> None:
        """freq_range_hz in auto-built SlotSpec equals (min, max) of
        context.evaluation_frequencies_hz — the full P06 verification band."""
        from unittest.mock import MagicMock

        from foster_eom.foster.schmidt import BranchRealization
        from foster_eom.optimize.variable_map import BranchCoordinates
        from foster_eom.realization.neighborhoods import build_slot_specs

        ctx = MagicMock()
        ctx.evaluation_frequencies_hz = (1e6, 5e6, 10e6, 20e6, 50e6)
        ctx.domain.branch1_realization = BranchRealization.FINITE_FOSTER
        ctx.domain.branch2_realization = BranchRealization.OPEN_OMITTED

        b1 = BranchCoordinates(
            k0=None,
            k_inf=None,
            k_residues=(1.0,),
            f_poles_hz=(10e6,),
            l_values_h=(1e-6,),
            c_values_f=(100e-12,),
        )
        b2 = BranchCoordinates(
            k0=None, k_inf=None, k_residues=(), f_poles_hz=(), l_values_h=(), c_values_f=()
        )

        specs = build_slot_specs(ctx, b1, b2)
        for s in specs:
            assert s.freq_range_hz == (1e6, 50e6), (
                f"Slot {s.element_id} has freq_range_hz={s.freq_range_hz}, expected (1e6, 50e6)"
            )

    def test_invariant_6_p06_verification_in_deb_order_top_k(self) -> None:
        """runner.py runs P06 verify on catalog_combos[:verify_top_k] (sorted by
        Deb key) — not best-only, not unsorted."""
        from pathlib import Path

        runner_src = (
            Path(__file__).parent.parent.parent / "foster_eom" / "realization" / "runner.py"
        ).read_text(encoding="utf-8")

        # Sort must happen BEFORE verification loop
        sort_pos = runner_src.index("catalog_combos.sort(key=lambda cc: cc.deb_key)")
        verify_pos = runner_src.index("for cc in catalog_combos[: spec.verify_top_k]:")
        assert sort_pos < verify_pos, "Deb sort must precede P06 verification loop"

        # Verification iterates the slice [:verify_top_k] of sorted list
        assert "catalog_combos[: spec.verify_top_k]" in runner_src

        # verified list accumulates all verified combos (not just passing ones)
        assert "verified.append(cc)" in runner_src
