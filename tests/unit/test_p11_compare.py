"""P11 compare.py unit tests.  No ngspice required."""

from __future__ import annotations

import math

import numpy as np
import pytest

from foster_eom.spice.compare import classify_status, compute_quantity_comparison
from foster_eom.spice.result import ValidationThresholds


def _thr():
    return ValidationThresholds()


def _freqs():
    return np.array([1e6, 2e6, 3e6, 4e6, 5e6], dtype=float)


class TestZeroError:
    def test_all_errors_zero(self):
        f = _freqs()
        v = np.array([1 + 0j, 2 + 1j, 3 + 2j, 4 + 3j, 5 + 4j])
        cmp = compute_quantity_comparison("X", f, v, v, _thr())
        assert cmp.max_abs_err == pytest.approx(0.0, abs=1e-30)
        assert cmp.max_rel_err == pytest.approx(0.0, abs=1e-30)
        assert cmp.max_phase_err_deg == pytest.approx(0.0, abs=1e-10)


class TestKnownMagnitudeError:
    def test_1pct_rel_error(self):
        f = _freqs()
        mna = np.array([100 + 0j] * 5)
        spice = mna * 1.01
        cmp = compute_quantity_comparison("Z", f, mna, spice, _thr())
        assert abs(cmp.max_rel_err - 0.01) < 1e-10


class TestKnownPhaseError:
    def test_phase_discrepancy(self):
        f = _freqs()
        mna = np.ones(5, dtype=complex) * (1 + 1j)
        phase_shift = 0.1  # radians
        spice = mna * np.exp(1j * phase_shift)
        cmp = compute_quantity_comparison("Z", f, mna, spice, _thr())
        expected_deg = phase_shift * 180.0 / math.pi
        assert abs(cmp.max_phase_err_deg - expected_deg) < 0.001


class TestSourceConventionRegression:
    """Regression: SPICE uses unit phasor; Python scales by vth_phasor.

    If vth_phasor = 2.0 + 0j, SPICE outputs are already at unit amplitude.
    After Python scales by 2.0, MNA and SPICE should agree exactly.
    This test confirms no hidden sqrt(2) factor is applied.
    """

    def test_unit_phasor_scale(self):
        vth = complex(2.0, 0.0)
        f = _freqs()
        # MNA: actual physical current (scaled by vth)
        mna_i = np.ones(5, dtype=complex) * (vth / 50.0)  # I = V/R
        # SPICE: unit-source current (before scaling)
        spice_unit_i = np.ones(5, dtype=complex) * (1.0 / 50.0)
        # Scale by vth_phasor in Python (as api.py does)
        spice_i = spice_unit_i * vth
        cmp = compute_quantity_comparison("I", f, mna_i, spice_i, _thr())
        assert cmp.max_abs_err < 1e-30
        # Confirm: if sqrt(2) had been applied, this would fail
        sqrt2_wrong = spice_unit_i * (abs(vth) * math.sqrt(2))
        cmp_wrong = compute_quantity_comparison("I_wrong", f, mna_i, sqrt2_wrong, _thr())
        assert cmp_wrong.max_rel_err > 0.4  # ~41% error if sqrt2 incorrectly applied


class TestCurrentDirection:
    """I(Vsense) > 0 into DUT; sign must be consistent with MNA i_port."""

    def test_current_sign_positive_into_dut(self):
        """If DUT voltage is positive and DUT is a resistor, current into DUT > 0."""
        f = _freqs()
        R = 100.0
        vth = complex(1.0, 0.0)
        Rs = 50.0
        # MNA: i_port = vth / (Rs + R) into DUT
        i_mna = vth / (Rs + R) * np.ones(5, dtype=complex)
        # SPICE sense: same, positive into DUT
        i_spice_unit = (1.0 / (Rs + R)) * np.ones(5, dtype=complex)
        i_spice = i_spice_unit * vth
        cmp = compute_quantity_comparison("I", f, i_mna, i_spice, _thr())
        assert cmp.max_rel_err < 1e-12
        # Both must be positive real
        assert np.all(np.real(i_mna) > 0)
        assert np.all(np.real(i_spice) > 0)


class TestPhaseMasking:
    def test_masked_when_mna_below_floor(self):
        thr = ValidationThresholds(mag_floor_for_phase=1.0)
        f = _freqs()
        mna = np.array([0.1 + 0j] * 5)  # all below floor=1.0
        spice = mna * np.exp(1j * 0.5)
        cmp = compute_quantity_comparison("Z", f, mna, spice, thr)
        assert cmp.n_phase_masked == 5
        assert math.isnan(cmp.max_phase_err_deg)


class TestResonance:
    def test_resonance_found(self):
        f = np.array([1e6, 2e6, 3e6, 4e6, 5e6], dtype=float)
        mna = np.array([1, 2, 10, 2, 1], dtype=complex)
        spice = np.array([1, 2, 9.5, 2, 1], dtype=complex)
        cmp = compute_quantity_comparison("Z", f, mna, spice, _thr(), compute_resonance=True)
        assert cmp.resonance_mna_hz == pytest.approx(3e6)
        assert cmp.resonance_spice_hz == pytest.approx(3e6)
        assert cmp.resonance_shift_hz == pytest.approx(0.0)


class TestClassifyStatus:
    def test_pass(self):
        f = _freqs()
        v = np.ones(5, dtype=complex)
        cmp = compute_quantity_comparison("Z", f, v, v, _thr())
        status, reason = classify_status([cmp], _thr())
        assert status == "pass"
        assert reason is None

    def test_fail(self):
        f = _freqs()
        mna = np.ones(5, dtype=complex)
        spice = mna * 2.0  # 100% error -> fail
        cmp = compute_quantity_comparison("Z", f, mna, spice, _thr())
        status, reason = classify_status([cmp], _thr())
        assert status == "fail"
        assert reason is not None
