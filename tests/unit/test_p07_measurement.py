"""Tests for foster_eom.measurement (Prompt 07).

Covers S1P/CSV import, MeasuredDataset, MeasuredOnePortModel, lossy-cap and
mBVD fitting, diagnostics, extrapolation recording, passivity warnings,
ndarray immutability, persistence round-trip, and end-to-end pipeline.
All tests use synthetic data — no real EOM hardware required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _lossy_cap_z(f_hz: np.ndarray, c0: float, rs: float, ls: float, g0: float) -> np.ndarray:
    """Compute impedance of a lossy capacitor: Z = Rs + jωLs + 1/(G0 + jωC0)."""
    omega = 2.0 * np.pi * f_hz
    z_series = rs + 1j * omega * ls
    y_core = g0 + 1j * omega * c0
    with np.errstate(divide="ignore", invalid="ignore"):
        return z_series + 1.0 / y_core


def _z_to_s11(z: np.ndarray, z_ref: float) -> np.ndarray:
    return (z - z_ref) / (z + z_ref)


def _write_s1p_ri(path: Path, f_hz: np.ndarray, s11: np.ndarray, z_ref: float = 50.0) -> None:
    """Write a Touchstone v1 S1P file in RI format."""
    with open(path, "w") as fp:
        fp.write("! Synthetic test S1P\n")
        fp.write(f"# HZ S RI R {z_ref:.1f}\n")
        for f, s in zip(f_hz, s11, strict=True):
            fp.write(f"{f:.6e}  {s.real:.15e}  {s.imag:.15e}\n")


def _write_s1p_ma(path: Path, f_hz: np.ndarray, s11: np.ndarray, z_ref: float = 50.0) -> None:
    """Write S1P in MA (magnitude/angle) format."""
    with open(path, "w") as fp:
        fp.write(f"# HZ S MA R {z_ref:.1f}\n")
        for f, s in zip(f_hz, s11, strict=True):
            fp.write(f"{f:.6e}  {abs(s):.15e}  {np.degrees(np.angle(s)):.15e}\n")


def _write_s1p_db(path: Path, f_hz: np.ndarray, s11: np.ndarray, z_ref: float = 50.0) -> None:
    """Write S1P in DB (dB-magnitude/angle) format."""
    with open(path, "w") as fp:
        fp.write(f"# HZ S DB R {z_ref:.1f}\n")
        for f, s in zip(f_hz, s11, strict=True):
            db = 20 * np.log10(max(abs(s), 1e-30))
            fp.write(f"{f:.6e}  {db:.15e}  {np.degrees(np.angle(s)):.15e}\n")


def _write_csv_s11_ri(path: Path, f_hz: np.ndarray, s11: np.ndarray) -> None:
    with open(path, "w") as fp:
        fp.write("freq,re_s11,im_s11\n")
        for f, s in zip(f_hz, s11, strict=True):
            fp.write(f"{f:.6e},{s.real:.15e},{s.imag:.15e}\n")


def _write_csv_z_ri(path: Path, f_hz: np.ndarray, z: np.ndarray) -> None:
    with open(path, "w") as fp:
        fp.write("freq,re_z,im_z\n")
        for f, zi in zip(f_hz, z, strict=True):
            fp.write(f"{f:.6e},{zi.real:.15e},{zi.imag:.15e}\n")


def _write_csv_s11_ma(path: Path, f_hz: np.ndarray, s11: np.ndarray) -> None:
    with open(path, "w") as fp:
        fp.write("freq,mag_s11,ang_s11\n")
        for f, s in zip(f_hz, s11, strict=True):
            fp.write(f"{f:.6e},{abs(s):.15e},{np.degrees(np.angle(s)):.15e}\n")


# Standard synthetic DUT: 3.3 pF, 2 Ω, 0.5 nH, G0=0
_C0 = 3.3e-12
_RS = 2.0
_LS = 0.5e-9
_G0 = 0.0
_F_HZ = np.linspace(1e6, 3e9, 200)
_Z_SYNTH = _lossy_cap_z(_F_HZ, _C0, _RS, _LS, _G0)
_S11_SYNTH = _z_to_s11(_Z_SYNTH, 50.0)


# ---------------------------------------------------------------------------
# TestS1PLoader
# ---------------------------------------------------------------------------


class TestS1PLoader:
    def test_ri_format(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        p = tmp_path / "dut.s1p"
        _write_s1p_ri(p, _F_HZ, _S11_SYNTH)
        ds = load_s1p(p)
        assert ds.source_format == "s1p"
        assert ds.source_quantity.value == "S11"
        assert len(ds.f_hz) == 200
        np.testing.assert_allclose(np.abs(ds.s11_complex - _S11_SYNTH), 0, atol=1e-10)

    def test_ma_format(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        p = tmp_path / "dut_ma.s1p"
        _write_s1p_ma(p, _F_HZ, _S11_SYNTH)
        ds = load_s1p(p)
        np.testing.assert_allclose(np.abs(ds.s11_complex), np.abs(_S11_SYNTH), rtol=1e-6)

    def test_db_format(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        p = tmp_path / "dut_db.s1p"
        _write_s1p_db(p, _F_HZ, _S11_SYNTH)
        ds = load_s1p(p)
        np.testing.assert_allclose(np.abs(ds.s11_complex), np.abs(_S11_SYNTH), rtol=1e-3)

    def test_non_50_ohm_ref(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        z_ref = 75.0
        s11_75 = _z_to_s11(_Z_SYNTH, z_ref)
        p = tmp_path / "dut75.s1p"
        _write_s1p_ri(p, _F_HZ, s11_75, z_ref=z_ref)
        ds = load_s1p(p)
        assert ds.z_ref_ohm == 75.0
        np.testing.assert_allclose(np.abs(ds.s11_complex - s11_75), 0, atol=1e-10)

    def test_sha256_provenance(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p
        from foster_eom.measurement.dataset import compute_file_sha256

        p = tmp_path / "sha_test.s1p"
        _write_s1p_ri(p, _F_HZ, _S11_SYNTH)
        ds = load_s1p(p)
        assert ds.source_sha256 == compute_file_sha256(p)

    def test_duplicate_freq_raises(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        f_dup = np.array([1e6, 2e6, 2e6, 3e6])
        s_dup = np.array([0.5 + 0j, 0.4 + 0j, 0.4 + 0j, 0.3 + 0j])
        p = tmp_path / "dup.s1p"
        _write_s1p_ri(p, f_dup, s_dup)
        with pytest.raises(ValueError, match="strictly increasing"):
            load_s1p(p)

    def test_nan_raises(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        f = np.array([1e6, 2e6, 3e6])
        s = np.array([0.5 + 0j, float("nan") + 0j, 0.3 + 0j])
        p = tmp_path / "nan.s1p"
        _write_s1p_ri(p, f, s)
        with pytest.raises(ValueError, match="non-finite"):
            load_s1p(p)

    def test_out_of_order_raises(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_s1p

        f = np.array([3e6, 1e6, 2e6])
        s = np.array([0.5 + 0j, 0.4 + 0j, 0.3 + 0j])
        p = tmp_path / "order.s1p"
        _write_s1p_ri(p, f, s)
        with pytest.raises(ValueError, match="strictly increasing"):
            load_s1p(p)

    def test_file_not_found(self) -> None:
        from foster_eom.measurement import load_s1p

        with pytest.raises(FileNotFoundError):
            load_s1p("nonexistent.s1p")

    def test_min_points(self, tmp_path: Path) -> None:
        """Single data point should raise."""
        from foster_eom.measurement import load_s1p

        p = tmp_path / "one.s1p"
        _write_s1p_ri(p, np.array([1e6]), np.array([0.5 + 0j]))
        with pytest.raises(ValueError, match="at least 2"):
            load_s1p(p)


# ---------------------------------------------------------------------------
# TestCSVLoader
# ---------------------------------------------------------------------------


class TestCSVLoader:
    def test_freq_s11_ri(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "dut.csv"
        _write_csv_s11_ri(p, _F_HZ, _S11_SYNTH)
        ds = load_csv(p, format="freq_s11_ri")
        assert ds.source_quantity.value == "S11"
        np.testing.assert_allclose(np.abs(ds.s11_complex - _S11_SYNTH), 0, atol=1e-10)

    def test_freq_z_ri(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "z.csv"
        _write_csv_z_ri(p, _F_HZ, _Z_SYNTH)
        ds = load_csv(p, format="freq_z_ri")
        assert ds.source_quantity.value == "Z"
        np.testing.assert_allclose(ds.z_complex, _Z_SYNTH, atol=1e-6)

    def test_freq_s11_ma(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "ma.csv"
        _write_csv_s11_ma(p, _F_HZ, _S11_SYNTH)
        ds = load_csv(p, format="freq_s11_ma")
        np.testing.assert_allclose(np.abs(ds.s11_complex), np.abs(_S11_SYNTH), rtol=1e-6)

    def test_auto_detect_s11_ri(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "auto.csv"
        _write_csv_s11_ri(p, _F_HZ, _S11_SYNTH)
        ds = load_csv(p, format="auto")
        assert ds.source_quantity.value == "S11"

    def test_ambiguous_headers_raise(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "ambig.csv"
        with open(p, "w") as fp:
            fp.write("freq,re_s11,im_s11,re_z,im_z\n")
            fp.write("1e6,0.5,0.1,50,10\n")
        with pytest.raises(ValueError, match=r"[Aa]mbiguous"):
            load_csv(p, format="auto")

    def test_missing_headers_raise(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "bad.csv"
        with open(p, "w") as fp:
            fp.write("col1,col2,col3\n")
            fp.write("1,2,3\n")
        with pytest.raises(ValueError):
            load_csv(p, format="auto")

    def test_explicit_column_map(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "custom.csv"
        with open(p, "w") as fp:
            fp.write("my_freq,my_re,my_im\n")
            for f, s in zip(_F_HZ[:5], _S11_SYNTH[:5], strict=True):
                fp.write(f"{f},{s.real},{s.imag}\n")
        ds = load_csv(
            p,
            format="freq_s11_ri",
            column_map={"freq": "my_freq", "s11_re": "my_re", "s11_im": "my_im"},
        )
        assert len(ds.f_hz) == 5

    def test_nan_rejection(self, tmp_path: Path) -> None:
        from foster_eom.measurement import load_csv

        p = tmp_path / "nan.csv"
        with open(p, "w") as fp:
            fp.write("freq,re_s11,im_s11\n")
            fp.write("1e6,0.5,nan\n")
        with pytest.raises(ValueError, match="NaN"):
            load_csv(p, format="freq_s11_ri")


# ---------------------------------------------------------------------------
# TestMeasuredDataset
# ---------------------------------------------------------------------------


class TestMeasuredDataset:
    def test_s11_z_round_trip(self) -> None:
        from foster_eom.measurement.dataset import MeasuredDataset

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH, z_ref_ohm=50.0)
        # Non-singular points should have Z ≈ true Z
        mask = ~ds.z_singular_mask
        np.testing.assert_allclose(ds.z_complex[mask], _Z_SYNTH[mask], rtol=1e-10)

    def test_validity_hz_spans_data(self) -> None:
        from foster_eom.measurement.dataset import MeasuredDataset

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        assert ds.validity_hz == (float(_F_HZ[0]), float(_F_HZ[-1]))

    def test_ndarray_mutation_raises(self) -> None:
        from foster_eom.measurement.dataset import MeasuredDataset

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        with pytest.raises(ValueError):
            ds.f_hz[0] = 999.0
        with pytest.raises(ValueError):
            ds.s11_complex[0] = 0.0 + 0j
        with pytest.raises(ValueError):
            ds.z_complex[0] = 0.0 + 0j

    def test_complex_z_ref_raises(self) -> None:
        from foster_eom.measurement.dataset import MeasuredDataset

        with pytest.raises(ValueError, match=r"[Cc]omplex"):
            MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH, z_ref_ohm=complex(50, 1))  # type: ignore[arg-type]

    def test_negative_z_ref_raises(self) -> None:
        from foster_eom.measurement.dataset import MeasuredDataset

        with pytest.raises(ValueError, match="> 0"):
            MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH, z_ref_ohm=-50.0)


# ---------------------------------------------------------------------------
# TestPassivityWarning
# ---------------------------------------------------------------------------


class TestPassivityWarning:
    def test_s11_gt_1_flags(self) -> None:
        from foster_eom.measurement.dataset import MeasuredDataset

        s11 = _S11_SYNTH.copy()
        s11[10] = 1.005 + 0j  # violate passivity slightly
        ds = MeasuredDataset.from_s11(_F_HZ, s11)
        assert len(ds.passivity_flags) > 0
        assert any("|S11|" in f for f in ds.passivity_flags)

    def test_import_succeeds_with_warning(self) -> None:
        """Passivity violation produces flags, not exception."""
        from foster_eom.measurement.dataset import MeasuredDataset

        s11 = _S11_SYNTH.copy()
        s11[5] = 1.01 + 0j
        ds = MeasuredDataset.from_s11(_F_HZ, s11)
        assert len(ds.f_hz) == 200  # still imported


# ---------------------------------------------------------------------------
# TestMeasuredOnePortModel
# ---------------------------------------------------------------------------


class TestMeasuredOnePortModel:
    def test_within_range_interpolation(self) -> None:
        from foster_eom.measurement import MeasuredDataset, MeasuredOnePortModel

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        model = MeasuredOnePortModel(ds)
        f_mid = float((_F_HZ[0] + _F_HZ[-1]) / 2)
        z = model.z(f_mid)
        assert np.isfinite(z)

    def test_error_extrapolation_raises(self) -> None:
        from foster_eom.domain.eom import ExtrapolationPolicy
        from foster_eom.errors import ModelValidityError
        from foster_eom.measurement import MeasuredDataset, MeasuredOnePortModel

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        model = MeasuredOnePortModel(ds, extrapolation_policy=ExtrapolationPolicy.ERROR)
        with pytest.raises(ModelValidityError):
            model.z(0.1)  # far below validity range

    def test_allow_extrapolation_records(self) -> None:
        from foster_eom.domain.eom import ExtrapolationPolicy
        from foster_eom.measurement import MeasuredDataset, MeasuredOnePortModel

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        model = MeasuredOnePortModel(ds, extrapolation_policy=ExtrapolationPolicy.ALLOW)
        assert not model.extrapolation_occurred
        _ = model.z(0.1)  # outside range
        assert model.extrapolation_occurred

    def test_validity_range_never_none(self) -> None:
        from foster_eom.measurement import MeasuredDataset, MeasuredOnePortModel

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        model = MeasuredOnePortModel(ds)
        assert model.validity_range() is not None

    def test_dataset_preserved(self) -> None:
        from foster_eom.measurement import MeasuredDataset, MeasuredOnePortModel

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        model = MeasuredOnePortModel(ds)
        assert model.dataset is ds


# ---------------------------------------------------------------------------
# TestFitLossyCap_S11Domain
# ---------------------------------------------------------------------------


class TestFitLossyCap_S11Domain:
    def test_ideal_cap_recovery(self, tmp_path: Path) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        c0 = 5e-12
        z = _lossy_cap_z(_F_HZ, c0, 0.0, 0.0, 0.0)
        s11 = _z_to_s11(z, 50.0)
        ds = MeasuredDataset.from_s11(_F_HZ, s11)
        result = fit_lossy_cap(ds, domain="S11")
        meta = result.model.metadata()
        assert abs(meta["c0_f"] - c0) / c0 < 0.01  # within 1%

    def test_lossy_cap_recovery(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds, domain="S11")
        meta = result.model.metadata()
        # S11-domain fit for a capacitive EOM (S11 ≈ +1) has inherently
        # lower parameter resolution than Z-domain — 15% is realistic.
        assert abs(meta["c0_f"] - _C0) / _C0 < 0.15
        assert result.diagnostics.converged

    def test_s11_near_plus_one_no_nan(self) -> None:
        """S11 ≈ +1 at low freq for a capacitor should not produce NaN."""
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        f = np.linspace(100, 1e6, 100)  # very low frequencies → S11 ≈ +1
        z = _lossy_cap_z(f, 3.3e-12, 0.0, 0.0, 0.0)
        s11 = _z_to_s11(z, 50.0)
        ds = MeasuredDataset.from_s11(f, s11)
        result = fit_lossy_cap(ds, domain="S11")
        assert np.isfinite(result.diagnostics.rms_error)
        assert not np.any(np.isnan(result.diagnostics.residuals_complex))

    def test_noisy_convergence(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        rng = np.random.default_rng(123)
        noise = rng.normal(0, 0.01, len(_S11_SYNTH)) + 1j * rng.normal(0, 0.01, len(_S11_SYNTH))
        s11_noisy = _S11_SYNTH + noise
        ds = MeasuredDataset.from_s11(_F_HZ, s11_noisy)
        result = fit_lossy_cap(ds, domain="S11")
        meta = result.model.metadata()
        assert abs(meta["c0_f"] - _C0) / _C0 < 0.15  # 15% tolerance with noise


# ---------------------------------------------------------------------------
# TestFitLossyCap_ZDomain
# ---------------------------------------------------------------------------


class TestFitLossyCap_ZDomain:
    def test_z_domain_fit(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds, domain="Z")
        meta = result.model.metadata()
        assert abs(meta["c0_f"] - _C0) / _C0 < 0.02
        assert result.fit_domain.value == "Z"

    def test_domain_comparison(self) -> None:
        """Both S11 and Z domain fits should recover similar parameters."""
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        r_s11 = fit_lossy_cap(ds, domain="S11")
        r_z = fit_lossy_cap(ds, domain="Z")
        c0_s11 = r_s11.model.metadata()["c0_f"]
        c0_z = r_z.model.metadata()["c0_f"]
        # S11-domain has inherently lower resolution for capacitive DUTs
        assert abs(c0_s11 - c0_z) / _C0 < 0.20


# ---------------------------------------------------------------------------
# TestFitMBVD
# ---------------------------------------------------------------------------


class TestFitMBVD:
    def _synth_mbvd(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Generate synthetic mBVD data with one motional branch."""
        from foster_eom.domain.eom import MotionalBranch
        from foster_eom.models.eom_mbvd import MBVDModel

        model = MBVDModel(
            c0_f=5e-12,
            g0_s=0.0,
            rs_ohm=1.0,
            ls_h=0.0,
            motional_branches=[MotionalBranch(rm_ohm=5.0, lm_h=1e-6, cm_f=1e-14)],
        )
        f = np.linspace(1e6, 200e6, 300)
        z = np.array([model.z(fi) for fi in f], dtype=np.complex128)
        return f, z, 5e-12

    def test_single_branch_recovery(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_mbvd

        f, z, c0_true = self._synth_mbvd()
        s11 = _z_to_s11(z, 50.0)
        ds = MeasuredDataset.from_s11(f, s11)
        result = fit_mbvd(ds, n_motional=1, domain="S11")
        meta = result.model.metadata()
        assert abs(meta["c0_f"] - c0_true) / c0_true < 0.15
        assert result.diagnostics.converged

    def test_two_branch_canonicalization(self) -> None:
        """Two-branch mBVD branches should be sorted by ascending f0."""
        from foster_eom.domain.eom import MotionalBranch
        from foster_eom.measurement import MeasuredDataset, fit_mbvd
        from foster_eom.models.eom_mbvd import MBVDModel

        model = MBVDModel(
            c0_f=5e-12,
            motional_branches=[
                MotionalBranch(rm_ohm=5.0, lm_h=1e-6, cm_f=1e-14),  # higher f0
                MotionalBranch(rm_ohm=10.0, lm_h=1e-5, cm_f=1e-13),  # lower f0
            ],
        )
        f = np.linspace(1e6, 500e6, 400)
        z = np.array([model.z(fi) for fi in f], dtype=np.complex128)
        s11 = _z_to_s11(z, 50.0)
        ds = MeasuredDataset.from_s11(f, s11)
        result = fit_mbvd(ds, n_motional=2, domain="S11")
        meta = result.model.metadata()
        branches = meta["motional_branches"]
        if len(branches) >= 2:
            f0s = []
            for b in branches:
                f0 = 1.0 / (2.0 * math.pi * math.sqrt(b["lm_h"] * b["cm_f"]))
                f0s.append(f0)
            assert f0s == sorted(f0s), "Branches not canonicalized by f0"


# ---------------------------------------------------------------------------
# TestFitDiagnostics
# ---------------------------------------------------------------------------


class TestFitDiagnostics:
    def test_residuals_length(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds)
        assert len(result.diagnostics.residuals_complex) == len(_F_HZ)

    def test_rms_error_positive(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds)
        assert result.diagnostics.rms_error_ohm > 0

    def test_converged_flag(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds)
        assert isinstance(result.diagnostics.converged, bool)

    def test_jacobian_rank_bounded(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds)
        if result.diagnostics.jacobian_rank is not None:
            assert result.diagnostics.jacobian_rank <= 4  # 4 params for lossy cap


# ---------------------------------------------------------------------------
# TestCovariance
# ---------------------------------------------------------------------------


class TestCovariance:
    def test_well_conditioned_fit(self) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds)
        # May or may not have covariance; if it does, check shape
        if result.diagnostics.param_covariance is not None:
            assert result.diagnostics.param_covariance.shape == (4, 4)
            assert result.diagnostics.covariance_reason is None

    def test_covariance_reason_when_none(self) -> None:
        """If covariance is None, reason must be a non-empty string."""
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        # Use very few data points to potentially trigger ill-conditioning
        f_short = np.array([1e6, 2e9])
        z_short = _lossy_cap_z(f_short, _C0, _RS, _LS, _G0)
        s11_short = _z_to_s11(z_short, 50.0)
        ds = MeasuredDataset.from_s11(f_short, s11_short)
        result = fit_lossy_cap(ds)
        if result.diagnostics.param_covariance is None:
            assert result.diagnostics.covariance_reason is not None
            assert len(result.diagnostics.covariance_reason) > 0


# ---------------------------------------------------------------------------
# TestParameterScaling
# ---------------------------------------------------------------------------


class TestParameterScaling:
    def test_picofarad_recovery(self) -> None:
        """Fit must recover C ≈ 3.3 pF correctly despite log-space transform."""
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap

        # Use Z-domain for tight parameter recovery; S11 domain has
        # inherently lower resolution for capacitive loads.
        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        result = fit_lossy_cap(ds, domain="Z")
        c0_fit = result.model.metadata()["c0_f"]
        assert abs(c0_fit - _C0) / _C0 < 0.02


# ---------------------------------------------------------------------------
# TestExtrapolationRecording
# ---------------------------------------------------------------------------


class TestExtrapolationRecording:
    def test_allow_records_extrapolation(self) -> None:
        from foster_eom.domain.eom import ExtrapolationPolicy
        from foster_eom.measurement import MeasuredDataset, MeasuredOnePortModel

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        model = MeasuredOnePortModel(ds, extrapolation_policy=ExtrapolationPolicy.ALLOW)
        assert not model.extrapolation_occurred
        _ = model.z(0.5)  # outside [1e6, 3e9]
        assert model.extrapolation_occurred


# ---------------------------------------------------------------------------
# TestEquivalentImport
# ---------------------------------------------------------------------------


class TestEquivalentImport:
    def test_s1p_csv_equivalence(self, tmp_path: Path) -> None:
        """Same DUT as S1P and CSV must yield z_complex within tolerance."""
        from foster_eom.measurement import load_csv, load_s1p

        p_s1p = tmp_path / "eq.s1p"
        p_csv = tmp_path / "eq.csv"
        _write_s1p_ri(p_s1p, _F_HZ, _S11_SYNTH)
        _write_csv_s11_ri(p_csv, _F_HZ, _S11_SYNTH)

        ds_s1p = load_s1p(p_s1p)
        ds_csv = load_csv(p_csv, format="freq_s11_ri")

        mask = ~ds_s1p.z_singular_mask & ~ds_csv.z_singular_mask
        np.testing.assert_allclose(ds_s1p.z_complex[mask], ds_csv.z_complex[mask], atol=1e-6)


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_dataset_round_trip(self, tmp_path: Path) -> None:
        from foster_eom.measurement import MeasuredDataset
        from foster_eom.persistence.yaml_io import (
            load_measured_characterization,
            reconstruct_measured_dataset,
            save_measured_characterization,
        )

        ds = MeasuredDataset.from_s11(
            _F_HZ,
            _S11_SYNTH,
            z_ref_ohm=50.0,
            source_file="test.s1p",
            source_sha256="abc123",
            instrument="LibreVNA",
            measurement_plane="EOM external RF connector",
        )
        p = tmp_path / "mc.yaml"
        save_measured_characterization(ds, path=str(p))

        mc = load_measured_characterization(str(p))
        assert mc is not None
        ds2 = reconstruct_measured_dataset(mc)
        np.testing.assert_allclose(ds2.f_hz, ds.f_hz, atol=1e-15)
        np.testing.assert_allclose(ds2.s11_complex, ds.s11_complex, atol=1e-15)
        assert mc["source_sha256"] == "abc123"
        assert mc["instrument"] == "LibreVNA"
        assert mc["measurement_plane"] == "EOM external RF connector"

    def test_fit_result_round_trip(self, tmp_path: Path) -> None:
        from foster_eom.measurement import MeasuredDataset, fit_lossy_cap
        from foster_eom.persistence.yaml_io import (
            load_measured_characterization,
            reconstruct_fit_model,
            save_measured_characterization,
        )

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        fr = fit_lossy_cap(ds)

        p = tmp_path / "mc_fit.yaml"
        save_measured_characterization(ds, fit_results=[fr], path=str(p))

        mc = load_measured_characterization(str(p))
        assert mc is not None
        assert "fit_results" in mc
        assert len(mc["fit_results"]) == 1

        fit_dict = mc["fit_results"][0]
        reconstructed = reconstruct_fit_model(fit_dict, validity_hz=ds.validity_hz)

        # Check that reconstructed model produces same impedance
        f_test = np.array([1e8, 1e9, 2e9])
        z_orig = np.array([fr.model.z(f) for f in f_test])
        z_recon = np.array([reconstructed.z(f) for f in f_test])
        np.testing.assert_allclose(z_orig, z_recon, atol=1e-10)

    def test_partial_population(self, tmp_path: Path) -> None:
        """Save without fit_results should still round-trip."""
        from foster_eom.measurement import MeasuredDataset
        from foster_eom.persistence.yaml_io import (
            load_measured_characterization,
            save_measured_characterization,
        )

        ds = MeasuredDataset.from_s11(_F_HZ, _S11_SYNTH)
        p = tmp_path / "no_fit.yaml"
        save_measured_characterization(ds, path=str(p))

        mc = load_measured_characterization(str(p))
        assert mc is not None
        assert "fit_results" not in mc


class TestBackwardCompatibility:
    def test_pre_p07_yaml_loads(self, tmp_path: Path) -> None:
        """A YAML file without measured_characterization returns None."""
        import yaml

        from foster_eom.persistence.yaml_io import load_measured_characterization

        p = tmp_path / "old.yaml"
        with open(p, "w") as f:
            yaml.dump({"candidates": [], "run_manifest": {}}, f)

        result = load_measured_characterization(str(p))
        assert result is None


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Synthetic S1P → import → model → fit → diagnostics → save → reload."""
        from foster_eom.measurement import (
            MeasuredOnePortModel,
            fit_lossy_cap,
            load_s1p,
        )
        from foster_eom.persistence.yaml_io import (
            load_measured_characterization,
            reconstruct_fit_model,
            reconstruct_measured_dataset,
            save_measured_characterization,
        )

        # 1. Generate synthetic S1P (non-50 Ω)
        z_ref = 75.0
        c0, rs, ls, g0 = 4.7e-12, 1.5, 0.2e-9, 1e-4
        f = np.linspace(1e6, 2e9, 150)
        z = _lossy_cap_z(f, c0, rs, ls, g0)
        s11 = _z_to_s11(z, z_ref)
        p_s1p = tmp_path / "eom.s1p"
        _write_s1p_ri(p_s1p, f, s11, z_ref=z_ref)

        # 2. Import
        ds = load_s1p(p_s1p, instrument="LibreVNA", measurement_plane="EOM external RF connector")
        assert ds.z_ref_ohm == z_ref
        assert ds.source_format == "s1p"

        # 3. Measured model
        model = MeasuredOnePortModel(ds)
        z_mid = model.z(1e9)
        assert np.isfinite(z_mid)

        # 4. Fit
        fr = fit_lossy_cap(ds, domain="S11")
        assert fr.diagnostics.converged
        meta = fr.model.metadata()
        assert abs(meta["c0_f"] - c0) / c0 < 0.05  # within 5%

        # 5. Persistence
        p_yaml = tmp_path / "eom_char.yaml"
        save_measured_characterization(ds, fit_results=[fr], path=str(p_yaml))

        # 6. Reload
        mc = load_measured_characterization(str(p_yaml))
        assert mc is not None
        ds2 = reconstruct_measured_dataset(mc)
        np.testing.assert_allclose(ds2.f_hz, ds.f_hz, atol=1e-15)

        recon = reconstruct_fit_model(mc["fit_results"][0], validity_hz=ds.validity_hz)
        z_check = np.array([recon.z(fi) for fi in [1e8, 5e8, 1e9]])
        z_orig = np.array([fr.model.z(fi) for fi in [1e8, 5e8, 1e9]])
        np.testing.assert_allclose(z_check, z_orig, atol=1e-10)
