"""Prompt-08 acceptance tests: component library backend.

~65 tests covering schema, file store, CRUD, model conditions, query,
Q semantics, model bridge, fallback policy, fixture extraction,
CSV importers, vendor adapters, Touchstone import, and persistence.

All tests use synthetic data and tmp_path — no internet dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(
    vendor: str = "TestVendor",
    part_number: str = "TEST-001",
    kind: str = "capacitor",
    value: float = 100e-12,
    **kwargs: object,
) -> Any:
    from foster_eom.catalog.component import ComponentKind, LibraryComponent

    c = LibraryComponent(
        id=str(uuid4()),
        kind=ComponentKind(kind),
        vendor=vendor,
        part_number=part_number,
        value_nom=value,
        import_source="test",
        import_ts="2024-01-01T00:00:00Z",
        **kwargs,  # type: ignore[arg-type]
    )
    c.content_sha256 = c.compute_content_sha256()
    return c


def _write_s1p_ri(path: Path, f_hz: np.ndarray, s11: np.ndarray, z_ref: float = 50.0) -> None:
    """Write a Touchstone v1 S1P file in RI format."""
    with open(path, "w") as fp:
        fp.write("! Synthetic test S1P\n")
        fp.write(f"# HZ S RI R {z_ref:.1f}\n")
        for f, s in zip(f_hz, s11, strict=True):
            fp.write(f"{f:.6e}  {s.real:.15e}  {s.imag:.15e}\n")


def _write_s2p_ri(
    path: Path,
    f_hz: np.ndarray,
    s_matrix: np.ndarray,
    z_ref: float = 50.0,
) -> None:
    """Write a Touchstone v1 S2P file in RI format.

    s_matrix: shape (n_freq, 2, 2)
    """
    with open(path, "w") as fp:
        fp.write(f"# HZ S RI R {z_ref:.1f}\n")
        for i in range(len(f_hz)):
            fp.write(
                f"{f_hz[i]:.6e}  "
                f"{s_matrix[i, 0, 0].real:.15e}  {s_matrix[i, 0, 0].imag:.15e}  "
                f"{s_matrix[i, 0, 1].real:.15e}  {s_matrix[i, 0, 1].imag:.15e}  "
                f"{s_matrix[i, 1, 0].real:.15e}  {s_matrix[i, 1, 0].imag:.15e}  "
                f"{s_matrix[i, 1, 1].real:.15e}  {s_matrix[i, 1, 1].imag:.15e}\n"
            )


def _ideal_inductor_z(f_hz: np.ndarray, l_h: float) -> np.ndarray:
    """Reference impedance for ideal inductor."""
    return 1j * 2.0 * np.pi * f_hz * l_h


def _shunt_dut_to_s2p(f_hz: np.ndarray, z_dut: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Construct 2-port S-parameters for a DUT in shunt configuration.

    Uses ABCD matrix approach (independent of extraction formula):
    Shunt impedance Z → ABCD = [[1, 0], [1/Z, 1]]
    Then convert ABCD → S.
    """
    n = len(f_hz)
    s = np.zeros((n, 2, 2), dtype=np.complex128)

    for i in range(n):
        # ABCD for shunt element
        a, b, c, d = 1.0, 0.0, 1.0 / z_dut[i], 1.0
        # ABCD → S conversion (standard formula)
        denom = a + b / z0 + c * z0 + d
        s[i, 0, 0] = (a + b / z0 - c * z0 - d) / denom
        s[i, 0, 1] = 2.0 * (a * d - b * c) / denom
        s[i, 1, 0] = 2.0 / denom
        s[i, 1, 1] = (-a + b / z0 - c * z0 + d) / denom

    return s


def _series_dut_to_s2p(f_hz: np.ndarray, z_dut: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Construct 2-port S-parameters for a DUT in series configuration.

    Uses ABCD matrix approach (independent of extraction formula):
    Series impedance Z → ABCD = [[1, Z], [0, 1]]
    """
    n = len(f_hz)
    s = np.zeros((n, 2, 2), dtype=np.complex128)

    for i in range(n):
        a, b, c, d = 1.0, z_dut[i], 0.0, 1.0
        denom = a + b / z0 + c * z0 + d
        s[i, 0, 0] = (a + b / z0 - c * z0 - d) / denom
        s[i, 0, 1] = 2.0 * (a * d - b * c) / denom
        s[i, 1, 0] = 2.0 / denom
        s[i, 1, 1] = (-a + b / z0 - c * z0 + d) / denom

    return s


# Standard test frequencies
_F_HZ = np.linspace(1e6, 100e6, 50)


# ===================================================================
# TestSchema
# ===================================================================


class TestSchema:
    def test_create_db(self, tmp_path: Path) -> None:
        from foster_eom.catalog.schema import create_db

        conn = create_db(tmp_path / "test.db")
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        assert cur.fetchone()[0] == 1
        conn.close()

    def test_version_row(self, tmp_path: Path) -> None:
        from foster_eom.catalog.schema import CURRENT_SCHEMA_VERSION, create_db

        conn = create_db(tmp_path / "test.db")
        cur = conn.execute("SELECT version FROM schema_version")
        assert cur.fetchone()[0] == CURRENT_SCHEMA_VERSION
        conn.close()

    def test_wal_mode(self, tmp_path: Path) -> None:
        from foster_eom.catalog.schema import check_wal_mode, create_db

        conn = create_db(tmp_path / "test.db")
        assert check_wal_mode(conn)
        conn.close()

    def test_idempotent_reopen(self, tmp_path: Path) -> None:
        from foster_eom.catalog.schema import create_db

        db = tmp_path / "test.db"
        conn1 = create_db(db)
        conn1.close()
        conn2 = create_db(db)
        cur = conn2.execute("SELECT MAX(version) FROM schema_version")
        assert cur.fetchone()[0] == 1
        conn2.close()


# ===================================================================
# TestFileStore
# ===================================================================


class TestFileStore:
    def test_store_retrieve(self, tmp_path: Path) -> None:
        from foster_eom.catalog.file_store import ContentAddressedStore

        store = ContentAddressedStore(tmp_path / "models")
        src = tmp_path / "test.s1p"
        src.write_text("test data")
        sha = store.store(src)
        assert len(sha) == 64
        retrieved = store.retrieve(sha, ".s1p")
        assert retrieved.read_text() == "test data"

    def test_verify(self, tmp_path: Path) -> None:
        from foster_eom.catalog.file_store import ContentAddressedStore

        store = ContentAddressedStore(tmp_path / "models")
        src = tmp_path / "test.s1p"
        src.write_text("verify me")
        sha = store.store(src)
        assert store.verify(sha, ".s1p")

    def test_dedup_noop(self, tmp_path: Path) -> None:
        from foster_eom.catalog.file_store import ContentAddressedStore

        store = ContentAddressedStore(tmp_path / "models")
        src = tmp_path / "test.s1p"
        src.write_text("same content")
        sha1 = store.store(src)
        sha2 = store.store(src)
        assert sha1 == sha2

    def test_missing_raises(self, tmp_path: Path) -> None:
        from foster_eom.catalog.file_store import ContentAddressedStore

        store = ContentAddressedStore(tmp_path / "models")
        with pytest.raises(FileNotFoundError):
            store.retrieve("abc123", ".s1p")

    def test_dir_structure(self, tmp_path: Path) -> None:
        from foster_eom.catalog.file_store import ContentAddressedStore

        store = ContentAddressedStore(tmp_path / "models")
        src = tmp_path / "test.s1p"
        src.write_text("dir test")
        sha = store.store(src)
        # Check that file is in sha[:2] subdirectory
        expected = tmp_path / "models" / sha[:2] / f"{sha}.s1p"
        assert expected.exists()

    def test_store_bytes(self, tmp_path: Path) -> None:
        from foster_eom.catalog.file_store import ContentAddressedStore

        store = ContentAddressedStore(tmp_path / "models")
        sha = store.store_bytes(b"byte data", ".s1p")
        p = store.retrieve(sha, ".s1p")
        assert p.read_bytes() == b"byte data"


# ===================================================================
# TestComponent
# ===================================================================


class TestComponent:
    def test_dataclass_construction(self) -> None:
        c = _make_component()
        assert c.vendor == "TestVendor"
        assert c.value_nom == 100e-12

    def test_component_kind_values(self) -> None:
        from foster_eom.catalog.component import ComponentKind

        assert ComponentKind.INDUCTOR.value == "inductor"
        assert ComponentKind.CAPACITOR.value == "capacitor"
        assert ComponentKind.RESISTOR.value == "resistor"

    def test_model_tier_ordering(self) -> None:
        from foster_eom.catalog.component import ModelTier, tier_rank

        assert tier_rank(ModelTier.MEASURED) > tier_rank(ModelTier.PARAMETRIC)
        assert tier_rank(ModelTier.PARAMETRIC) > tier_rank(ModelTier.IDEAL)

    def test_model_origin_values(self) -> None:
        from foster_eom.catalog.component import ModelOrigin

        assert ModelOrigin.VENDOR_TOUCHSTONE.value == "vendor_touchstone"
        assert ModelOrigin.IDEAL.value == "ideal"


# ===================================================================
# TestCRUD
# ===================================================================


class TestCRUD:
    def test_add_get(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            got = lib.get(cid)
            assert got.part_number == "TEST-001"

    def test_get_by_part(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            lib.add(_make_component())
            got = lib.get_by_part("TestVendor", "TEST-001")
            assert got.value_nom == 100e-12

    def test_delete(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            lib.delete(cid)
            with pytest.raises(KeyError):
                lib.get(cid)

    def test_idempotent_reimport(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            lib.add(c)
            cid2 = lib.add(c)
            assert cid2 == "__skipped_dup__"

    def test_error_on_mismatch(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            lib.add(_make_component(value=100e-12))
            with pytest.raises(ValueError, match=r"already exists"):
                lib.add(_make_component(value=200e-12))

    def test_merge(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            lib.add(_make_component(value=100e-12))
            c2 = _make_component(value=200e-12, package="0402")
            c2.content_sha256 = c2.compute_content_sha256()
            lib.add(c2, on_conflict="merge")
            got = lib.get_by_part("TestVendor", "TEST-001")
            assert got.package == "0402"

    def test_replace(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            lib.add(_make_component(value=100e-12))
            c2 = _make_component(value=200e-12)
            c2.content_sha256 = c2.compute_content_sha256()
            lib.add(c2, on_conflict="replace")
            got = lib.get_by_part("TestVendor", "TEST-001")
            assert got.value_nom == 200e-12

    def test_content_sha256_stable(self) -> None:
        c1 = _make_component()
        c2 = _make_component()
        assert c1.content_sha256 == c2.content_sha256


# ===================================================================
# TestModelConditions
# ===================================================================


class TestModelConditions:
    def test_add_and_retrieve(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            mc = ModelCondition(
                id=str(uuid4()),
                component_id=cid,
                model_tier=ModelTier.PARAMETRIC,
                model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                parametric_params={"esr_ohm": 0.05},
                import_ts="2024-01-01T00:00:00Z",
            )
            lib.add_model_condition(mc)
            conditions = lib.get_model_conditions(cid)
            # ideal + parametric = 2
            assert len(conditions) >= 2

    def test_multiple_same_tier(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            # Two measured models with different frequency spans
            for label, lo, hi in [("low", 1e6, 50e6), ("high", 50e6, 200e6)]:
                mc = ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.MEASURED,
                    model_origin=ModelOrigin.LAB_MEASUREMENT,
                    validity_hz_lo=lo,
                    validity_hz_hi=hi,
                    variant_label=label,
                    import_ts="2024-01-01T00:00:00Z",
                )
                lib.add_model_condition(mc)
            conditions = lib.get_model_conditions(cid)
            measured = [c for c in conditions if c.model_tier == ModelTier.MEASURED]
            assert len(measured) == 2

    def test_cascade_delete(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            lib.delete(cid)
            conditions = lib.get_model_conditions(cid)
            assert len(conditions) == 0

    def test_variant_label(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            mc = ModelCondition(
                id=str(uuid4()),
                component_id=cid,
                model_tier=ModelTier.MEASURED,
                model_origin=ModelOrigin.LAB_MEASUREMENT,
                variant_label="revision_B",
                import_ts="2024-01-01T00:00:00Z",
            )
            lib.add_model_condition(mc)
            conditions = lib.get_model_conditions(cid)
            labels = [c.variant_label for c in conditions]
            assert "revision_B" in labels

    def test_parametric_params_roundtrip(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="inductor", value=47e-9)
            cid = lib.add(c)
            params = {"dcr_ohm": 0.15, "c_par_f": 2.3e-12}
            mc = ModelCondition(
                id=str(uuid4()),
                component_id=cid,
                model_tier=ModelTier.PARAMETRIC,
                model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                parametric_params=params,
                import_ts="2024-01-01T00:00:00Z",
            )
            lib.add_model_condition(mc)
            conditions = lib.get_model_conditions(cid)
            parametric = [c for c in conditions if c.model_tier == ModelTier.PARAMETRIC]
            assert len(parametric) >= 1
            assert parametric[0].parametric_params["dcr_ohm"] == pytest.approx(0.15)


# ===================================================================
# TestQuery
# ===================================================================


class TestQuery:
    def _populate(self, lib: Any) -> None:
        """Add a few components for query testing."""
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier

        # C1: 100pF cap, 25V, 0402, 5%, ESR=0.05
        c1 = _make_component(
            part_number="C1",
            value=100e-12,
            voltage_max_v=25.0,
            package="0402",
            value_tol_frac=0.05,
        )
        cid1 = lib.add(c1)
        lib.add_model_condition(
            ModelCondition(
                id=str(uuid4()),
                component_id=cid1,
                model_tier=ModelTier.PARAMETRIC,
                model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                esr_ohm=0.05,
                srf_hz=500e6,
                q_value=200,
                q_at_f_hz=100e6,
                validity_hz_lo=1e6,
                validity_hz_hi=400e6,
                import_ts="2024-01-01T00:00:00Z",
            )
        )

        # C2: 10nF cap, 50V, 0805, 10%
        c2 = _make_component(
            part_number="C2",
            value=10e-9,
            voltage_max_v=50.0,
            package="0805",
            value_tol_frac=0.10,
        )
        lib.add(c2)

        # L1: 47nH inductor, in stock
        l1 = _make_component(
            part_number="L1",
            kind="inductor",
            value=47e-9,
            stock_status="in_stock",
            stock_ts="2024-06-01T00:00:00Z",
        )
        lib.add(l1)

    def test_query_kind(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery
        from foster_eom.catalog.component import ComponentKind

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(kind=ComponentKind.CAPACITOR))
            assert len(results) == 2

    def test_query_value_range(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(value_min=50e-12, value_max=200e-12))
            assert len(results) == 1
            assert results[0].part_number == "C1"

    def test_query_tol(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(tol_max_frac=0.05))
            assert len(results) == 1

    def test_query_voltage(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(voltage_min_v=40.0))
            assert len(results) == 1
            assert results[0].part_number == "C2"

    def test_query_srf(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(srf_min_hz=400e6))
            assert len(results) == 1

    def test_query_esr(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(esr_max_ohm=0.1))
            assert len(results) == 1

    def test_query_freq_range(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(freq_range_hz=(10e6, 300e6)))
            assert len(results) == 1
            assert results[0].part_number == "C1"

    def test_query_in_stock(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(in_stock_only=True))
            assert len(results) == 1
            assert results[0].part_number == "L1"

    def test_query_glob(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(part_number_glob="C%"))
            assert len(results) == 2

    def test_query_empty(self, tmp_path: Path) -> None:
        from foster_eom.catalog import ComponentLibrary, ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            self._populate(lib)
            results = lib.query(ComponentQuery(value_min=1.0))  # 1 Farad
            assert len(results) == 0


# ===================================================================
# TestQSemantics
# ===================================================================


class TestQSemantics:
    def test_q_with_freq(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary
        from foster_eom.catalog.query import ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.PARAMETRIC,
                    model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                    q_value=150,
                    q_at_f_hz=50e6,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            results = lib.query(ComponentQuery(q_min=100))
            assert len(results) == 1

    def test_q_without_freq_excluded(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary
        from foster_eom.catalog.query import ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            # Q value without q_at_f_hz should NOT match q_min query
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.PARAMETRIC,
                    model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                    q_value=150,
                    q_at_f_hz=None,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            results = lib.query(ComponentQuery(q_min=100))
            assert len(results) == 0

    def test_q_min_filter(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary
        from foster_eom.catalog.query import ComponentQuery

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.PARAMETRIC,
                    model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                    q_value=50,
                    q_at_f_hz=100e6,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            # Q=50 should NOT match q_min=100
            results = lib.query(ComponentQuery(q_min=100))
            assert len(results) == 0


# ===================================================================
# TestModelBridge
# ===================================================================


class TestModelBridge:
    def test_ideal_capacitor(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="capacitor", value=100e-12)
            cid = lib.add(c)
            model = lib.build_model(cid)
            z = model.z(100e6)
            # Z of ideal 100pF at 100 MHz
            expected = 1.0 / (1j * 2 * np.pi * 100e6 * 100e-12)
            assert abs(z - expected) < 1e-3

    def test_ideal_inductor(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="inductor", value=47e-9)
            cid = lib.add(c)
            model = lib.build_model(cid)
            z = model.z(100e6)
            expected = 1j * 2 * np.pi * 100e6 * 47e-9
            assert abs(z - expected) < 1e-3

    def test_ideal_resistor(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="resistor", value=100.0)
            cid = lib.add(c)
            model = lib.build_model(cid)
            z = model.z(100e6)
            assert abs(z - 100.0) < 1e-6

    def test_parametric_inductor(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="inductor", value=47e-9)
            cid = lib.add(c)
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.PARAMETRIC,
                    model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                    parametric_params={"dcr_ohm": 0.5, "c_par_f": 0.0},
                    validity_hz_lo=1e6,
                    validity_hz_hi=500e6,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            from foster_eom.catalog.component import FallbackPolicy

            model = lib.build_model(cid, fallback=FallbackPolicy.ALLOW_LOWER_TIER)
            # Parametric should be selected over ideal
            z = model.z(100e6)
            # LumpedLossyInductor(47nH, DCR=0.5)
            expected = 0.5 + 1j * 2 * np.pi * 100e6 * 47e-9
            assert abs(z - expected) / abs(expected) < 0.01

    def test_parametric_capacitor(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelCondition, ModelOrigin, ModelTier
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="capacitor", value=100e-12)
            cid = lib.add(c)
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.PARAMETRIC,
                    model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                    parametric_params={"esr_ohm": 0.1, "esl_h": 0.0},
                    validity_hz_lo=1e6,
                    validity_hz_hi=500e6,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            from foster_eom.catalog.component import FallbackPolicy

            model = lib.build_model(cid, fallback=FallbackPolicy.ALLOW_LOWER_TIER)
            z = model.z(100e6)
            expected = 0.1 + 1.0 / (1j * 2 * np.pi * 100e6 * 100e-12)
            assert abs(z - expected) / abs(expected) < 0.01


# ===================================================================
# TestFallbackPolicy
# ===================================================================


class TestFallbackPolicy:
    def test_strict_blocks_when_higher_exists(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import (
            FallbackPolicy,
            ModelCondition,
            ModelOrigin,
            ModelTier,
        )
        from foster_eom.catalog.library import ComponentLibrary
        from foster_eom.catalog.model_bridge import ModelNotAvailableError

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="inductor", value=47e-9)
            cid = lib.add(c)
            # Add measured model valid only 1-50 MHz
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.MEASURED,
                    model_origin=ModelOrigin.LAB_MEASUREMENT,
                    validity_hz_lo=1e6,
                    validity_hz_hi=50e6,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            # Request 1-200 MHz → measured doesn't cover, STRICT blocks ideal
            with pytest.raises(ModelNotAvailableError, match=r"STRICT"):
                lib.build_model(
                    cid,
                    freq_range=(1e6, 200e6),
                    fallback=FallbackPolicy.STRICT,
                )

    def test_allow_lower_tier(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import (
            FallbackPolicy,
            ModelCondition,
            ModelOrigin,
            ModelTier,
        )
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(kind="inductor", value=47e-9)
            cid = lib.add(c)
            lib.add_model_condition(
                ModelCondition(
                    id=str(uuid4()),
                    component_id=cid,
                    model_tier=ModelTier.MEASURED,
                    model_origin=ModelOrigin.LAB_MEASUREMENT,
                    validity_hz_lo=1e6,
                    validity_hz_hi=50e6,
                    import_ts="2024-01-01T00:00:00Z",
                )
            )
            # ALLOW_LOWER_TIER should fall back to ideal
            model = lib.build_model(
                cid,
                freq_range=(1e6, 200e6),
                fallback=FallbackPolicy.ALLOW_LOWER_TIER,
            )
            assert model is not None

    def test_required_tier(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import ModelTier
        from foster_eom.catalog.library import ComponentLibrary
        from foster_eom.catalog.model_bridge import ModelNotAvailableError

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component()
            cid = lib.add(c)
            with pytest.raises(ModelNotAvailableError, match=r"measured"):
                lib.build_model(cid, required_tier=ModelTier.MEASURED)

    def test_no_model_error(self, tmp_path: Path) -> None:
        from foster_eom.catalog.component import (
            ComponentKind,
            FallbackPolicy,
            LibraryComponent,
        )
        from foster_eom.catalog.model_bridge import ModelNotAvailableError, build_model

        comp = LibraryComponent(
            id="fake",
            kind=ComponentKind.CAPACITOR,
            vendor="V",
            part_number="P",
            value_nom=1e-12,
            import_source="test",
            import_ts="2024-01-01T00:00:00Z",
        )
        with pytest.raises(ModelNotAvailableError, match=r"No model conditions"):
            build_model(comp, [], fallback=FallbackPolicy.STRICT)


# ===================================================================
# TestFixture — independently validated S2P extraction
# ===================================================================


class TestFixture:
    def test_s1p_direct(self, tmp_path: Path) -> None:
        """S1P file → TabularImpedanceComponent, no fixture needed."""
        from foster_eom.catalog.fixture import extract_one_port

        # Synthetic 50Ω S1P (S11≈0 for matched load)
        f = _F_HZ
        z_ref = 50.0
        z_load = np.full_like(f, 100.0 + 0j)
        s11 = (z_load - z_ref) / (z_load + z_ref)
        p = tmp_path / "load.s1p"
        _write_s1p_ri(p, f, s11, z_ref=z_ref)

        model = extract_one_port(p)
        z = model.z(f)
        np.testing.assert_allclose(z.real, 100.0, rtol=1e-6)

    def test_s2p_shunt_independent(self, tmp_path: Path) -> None:
        """S2P shunt fixture: independently constructed via ABCD, extracted,
        and compared against known DUT impedance.

        The test constructs the S2P using ABCD matrices (transfer matrix
        approach) and extracts using the Y-parameter formula — these are
        independent methods, verifying correctness.
        """
        from foster_eom.catalog.fixture import FixtureSpec, FixtureType, extract_one_port

        l_h = 47e-9
        f = _F_HZ
        z_dut = _ideal_inductor_z(f, l_h)

        # Generate S2P via ABCD (independent of extraction formula)
        s2p = _shunt_dut_to_s2p(f, z_dut)
        p = tmp_path / "shunt_l.s2p"
        _write_s2p_ri(p, f, s2p)

        fixture = FixtureSpec(fixture_type=FixtureType.SHUNT, port_z=0, port_gnd=1)
        model = extract_one_port(p, fixture)
        z_recovered = model.z(f)

        np.testing.assert_allclose(z_recovered.real, z_dut.real, atol=1e-3)
        np.testing.assert_allclose(z_recovered.imag, z_dut.imag, rtol=1e-4)

    def test_s2p_series_independent(self, tmp_path: Path) -> None:
        """S2P series fixture: independently constructed via ABCD."""
        from foster_eom.catalog.fixture import FixtureSpec, FixtureType, extract_one_port

        l_h = 47e-9
        f = _F_HZ
        z_dut = _ideal_inductor_z(f, l_h)

        s2p = _series_dut_to_s2p(f, z_dut)
        p = tmp_path / "series_l.s2p"
        _write_s2p_ri(p, f, s2p)

        fixture = FixtureSpec(fixture_type=FixtureType.SERIES, port_z=0, port_gnd=1)
        model = extract_one_port(p, fixture)
        z_recovered = model.z(f)

        np.testing.assert_allclose(z_recovered.real, z_dut.real, atol=1e-3)
        np.testing.assert_allclose(z_recovered.imag, z_dut.imag, rtol=1e-4)

    def test_invalid_fixture_raises(self, tmp_path: Path) -> None:
        """Multiport without fixture raises ValueError."""
        from foster_eom.catalog.fixture import extract_one_port

        l_h = 47e-9
        f = _F_HZ
        z_dut = _ideal_inductor_z(f, l_h)
        s2p = _shunt_dut_to_s2p(f, z_dut)
        p = tmp_path / "no_fixture.s2p"
        _write_s2p_ri(p, f, s2p)

        with pytest.raises(ValueError, match=r"FixtureSpec required"):
            extract_one_port(p)


# ===================================================================
# TestCSVImport
# ===================================================================


class TestCSVImport:
    def test_generic_import(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "parts.csv"
        csv_path.write_text(
            "vendor,part_number,kind,value,unit,tolerance\n"
            "TestCo,TC-100,capacitor,100pF,,5%\n"
            "TestCo,TC-200,inductor,47nH,,10%\n"
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            result = lib.import_csv(csv_path)
            assert result.inserted == 2
            assert result.skipped_error == 0

    def test_si_prefix_parsing(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "parts.csv"
        csv_path.write_text(
            "vendor,part_number,kind,value\n"
            "TestCo,TC-1,capacitor,100pF\n"
            "TestCo,TC-2,inductor,47nH\n"
            "TestCo,TC-3,resistor,2.2kohm\n"
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            lib.import_csv(csv_path)
            c1 = lib.get_by_part("TestCo", "TC-1")
            assert c1.value_nom == pytest.approx(100e-12)
            c2 = lib.get_by_part("TestCo", "TC-2")
            assert c2.value_nom == pytest.approx(47e-9)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "parts.csv"
        csv_path.write_text(
            "vendor,kind,value\nTestCo,capacitor,100pF\n"  # missing part_number
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            result = lib.import_csv(csv_path)
            assert result.skipped_error == 1
            assert len(result.errors) == 1

    def test_tolerance_parsing(self, tmp_path: Path) -> None:
        from foster_eom.catalog.importers.csv_generic import parse_tolerance

        assert parse_tolerance("5%") == pytest.approx(0.05)
        assert parse_tolerance("±2%") == pytest.approx(0.02)
        assert parse_tolerance("0.1") == pytest.approx(0.1)
        assert parse_tolerance("") is None


# ===================================================================
# TestVendorAdapters
# ===================================================================


class TestVendorAdapters:
    def test_murata_header_parsing(self, tmp_path: Path) -> None:
        from foster_eom.catalog.importers.csv_murata import MurataCSVImporter
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "murata.csv"
        csv_path.write_text(
            "Part Number,Capacitance,Tolerance,Rated Voltage,Size,Temperature Characteristic,ESR\n"
            "GRM155R71C104KA88,100nF,10%,16V,0402,X7R,0.05\n"
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            importer = MurataCSVImporter()
            result = importer.import_to(lib, csv_path)
            assert result.inserted == 1
            c = lib.get_by_part("Murata", "GRM155R71C104KA88")
            assert c.package == "0402"

    def test_coilcraft_header_parsing(self, tmp_path: Path) -> None:
        from foster_eom.catalog.importers.csv_coilcraft import CoilcraftCSVImporter
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "coilcraft.csv"
        csv_path.write_text(
            "Part Number,Inductance,Tolerance,DCR Typ,SRF Min,Irms,Isat,Size,Series\n"
            "XAL5030-472M,4.7uH,20%,0.035,50MHz,5.5,8.2,XAL5030,XAL\n"
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            importer = CoilcraftCSVImporter()
            result = importer.import_to(lib, csv_path)
            assert result.inserted == 1
            c = lib.get_by_part("Coilcraft", "XAL5030-472M")
            assert c.value_nom == pytest.approx(4.7e-6)

    def test_csv_columns_primary(self, tmp_path: Path) -> None:
        """Verify that explicit CSV columns override part-number decoding."""
        from foster_eom.catalog.importers.csv_murata import MurataCSVImporter
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "murata_override.csv"
        csv_path.write_text(
            "Part Number,Capacitance,Tolerance,Rated Voltage,Size,Temperature Characteristic\n"
            "GRM155R71C104KA88,220nF,5%,25V,0603,X5R\n"
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            importer = MurataCSVImporter()
            importer.import_to(lib, csv_path)
            c = lib.get_by_part("Murata", "GRM155R71C104KA88")
            # CSV says 220nF and 0603, not the decoded 100nF and 0402
            assert c.value_nom == pytest.approx(220e-9)
            assert c.package == "0603"

    def test_coilcraft_srf_to_cpar(self, tmp_path: Path) -> None:
        """Verify SRF→Cpar derivation."""
        from foster_eom.catalog.importers.csv_coilcraft import CoilcraftCSVImporter
        from foster_eom.catalog.library import ComponentLibrary

        csv_path = tmp_path / "coilcraft_srf.csv"
        csv_path.write_text(
            "Part Number,Inductance,DCR Typ,SRF Min\nXAL5030-472M,4.7uH,0.035,50MHz\n"
        )
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            importer = CoilcraftCSVImporter()
            importer.import_to(lib, csv_path)
            c = lib.get_by_part("Coilcraft", "XAL5030-472M")
            conditions = lib.get_model_conditions(c.id)
            parametric = [mc for mc in conditions if mc.model_tier.value == "parametric"]
            assert len(parametric) >= 1
            params = parametric[0].parametric_params
            assert params is not None
            # Cpar = 1/((2π·50e6)²·4.7e-6) ≈ 2.15 pF
            expected = 1.0 / ((2.0 * math.pi * 50e6) ** 2 * 4.7e-6)
            assert params["c_par_f"] == pytest.approx(expected, rel=0.01)


# ===================================================================
# TestTouchstoneImport
# ===================================================================


class TestTouchstoneImport:
    def test_bulk_import(self, tmp_path: Path) -> None:
        from foster_eom.catalog.importers.touchstone import TouchstoneImporter
        from foster_eom.catalog.library import ComponentLibrary

        # Create component first
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(vendor="TestCo", part_number="IND-47N")
            lib.add(c)

            # Create S1P directory
            ts_dir = tmp_path / "touchstone"
            ts_dir.mkdir()
            f = _F_HZ
            s11 = np.zeros_like(f, dtype=complex)  # matched load
            _write_s1p_ri(ts_dir / "IND-47N.s1p", f, s11)

            imp = TouchstoneImporter(vendor="TestCo")
            result = imp.import_to(lib, ts_dir)
            assert result.inserted == 1

            # Verify model condition was created
            conditions = lib.get_model_conditions(c.id)
            measured = [mc for mc in conditions if mc.model_tier.value == "measured"]
            assert len(measured) == 1
            assert measured[0].model_file_sha256 is not None

    def test_fixture_required_for_s2p(self, tmp_path: Path) -> None:
        from foster_eom.catalog.importers.touchstone import TouchstoneImporter
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            c = _make_component(vendor="TestCo", part_number="IND-47N")
            lib.add(c)

            ts_dir = tmp_path / "touchstone"
            ts_dir.mkdir()
            f = _F_HZ
            z_dut = _ideal_inductor_z(f, 47e-9)
            s2p = _shunt_dut_to_s2p(f, z_dut)
            _write_s2p_ri(ts_dir / "IND-47N.s2p", f, s2p)

            imp = TouchstoneImporter(vendor="TestCo")
            result = imp.import_to(lib, ts_dir)
            # Should fail because no fixture provided for s2p
            assert result.skipped_error == 1

    def test_sha_verified(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            f = _F_HZ
            s11 = np.zeros_like(f, dtype=complex)
            p = tmp_path / "test.s1p"
            _write_s1p_ri(p, f, s11)

            sha = lib.file_store.store(p)
            assert lib.file_store.verify(sha, ".s1p")


# ===================================================================
# TestPersistence
# ===================================================================


class TestPersistence:
    def test_library_sha_determinism(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        with ComponentLibrary(tmp_path / "lib.db") as lib:
            lib.add(_make_component(part_number="P1", value=1e-12))
            lib.add(_make_component(part_number="P2", value=2e-12))
            sha1 = lib.library_sha256()

        # Re-open and compute again
        with ComponentLibrary(tmp_path / "lib.db") as lib:
            sha2 = lib.library_sha256()

        assert sha1 == sha2

    def test_sha_stability_across_reopen(self, tmp_path: Path) -> None:
        from foster_eom.catalog.library import ComponentLibrary

        db = tmp_path / "lib.db"
        with ComponentLibrary(db) as lib:
            lib.add(_make_component())
            sha1 = lib.library_sha256()

        with ComponentLibrary(db) as lib:
            sha2 = lib.library_sha256()

        assert sha1 == sha2

    def test_relative_path_yaml_roundtrip(self, tmp_path: Path) -> None:
        from foster_eom.persistence.yaml_io import load_library_ref, save_library_ref

        project = tmp_path / "project.fseom.yaml"
        lib_path = tmp_path / "components" / "library.fseom.db"
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.touch()

        save_library_ref(project, lib_path, "abc123def456")
        ref = load_library_ref(project)
        assert ref is not None
        assert "components/library.fseom.db" in ref["path"]
        assert ref["sha256"] == "abc123def456"

    def test_path_hash_mismatch_loads(self, tmp_path: Path) -> None:
        """Library ref with wrong SHA should still load (warning only)."""
        from foster_eom.persistence.yaml_io import load_library_ref, save_library_ref

        project = tmp_path / "project.fseom.yaml"
        lib_path = tmp_path / "lib.db"
        lib_path.touch()

        save_library_ref(project, lib_path, "wrong_sha")
        ref = load_library_ref(project)
        assert ref is not None
        assert ref["sha256"] == "wrong_sha"
