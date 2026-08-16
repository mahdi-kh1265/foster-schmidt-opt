"""Tests for foster_eom.units.

Covers frequency, inductance, capacitance, voltage, current, power, angle
conversions, source-convention transforms (Thévenin ↔ available power),
and impedance/reflection calculations.

Particular attention to sqrt(2) and factor-of-two traps per Prompt 01.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from foster_eom.units import (
    SQRT2,
    available_power_to_vth_rms,
    dbm_to_w,
    deg_to_rad,
    f_to_nf,
    f_to_pf,
    gamma_to_z,
    generator_display_to_vth_rms,
    ghz_to_hz,
    h_to_nh,
    h_to_uh,
    hz_to_ghz,
    hz_to_khz,
    hz_to_mhz,
    hz_to_rad_per_s,
    ipeak_to_irms,
    irms_to_ipeak,
    khz_to_hz,
    mhz_to_hz,
    nf_to_f,
    nh_to_h,
    pf_to_f,
    rad_per_s_to_hz,
    rad_to_deg,
    s11_db,
    s11_db_array,
    s11_db_from_gamma,
    uh_to_h,
    vpeak_to_vrms,
    vpp_to_vrms,
    vrms_to_vpeak,
    vrms_to_vpp,
    vth_rms_to_available_power,
    w_to_dbm,
    z_to_gamma,
    z_to_gamma_array,
)

# ---------------------------------------------------------------------------
# Frequency conversions
# ---------------------------------------------------------------------------

class TestFrequencyConversions:
    def test_khz_round_trip(self) -> None:
        assert hz_to_khz(khz_to_hz(1.0)) == pytest.approx(1.0)

    def test_mhz_round_trip(self) -> None:
        assert hz_to_mhz(mhz_to_hz(10.0)) == pytest.approx(10.0)

    def test_ghz_round_trip(self) -> None:
        assert hz_to_ghz(ghz_to_hz(2.4)) == pytest.approx(2.4)

    def test_mhz_to_hz_value(self) -> None:
        assert mhz_to_hz(10.0) == pytest.approx(10.0e6)

    def test_hz_to_rad_per_s(self) -> None:
        assert hz_to_rad_per_s(1.0) == pytest.approx(2.0 * math.pi)

    def test_rad_per_s_round_trip(self) -> None:
        assert rad_per_s_to_hz(hz_to_rad_per_s(100.0)) == pytest.approx(100.0)

    @given(st.floats(min_value=1e-3, max_value=1e15))
    def test_mhz_round_trip_hypothesis(self, f: float) -> None:
        assert hz_to_mhz(mhz_to_hz(f)) == pytest.approx(f, rel=1e-12)


# ---------------------------------------------------------------------------
# Inductance conversions
# ---------------------------------------------------------------------------

class TestInductanceConversions:
    def test_uh_round_trip(self) -> None:
        assert h_to_uh(uh_to_h(4.7)) == pytest.approx(4.7)

    def test_nh_round_trip(self) -> None:
        assert h_to_nh(nh_to_h(100.0)) == pytest.approx(100.0)

    def test_nh_to_h_value(self) -> None:
        assert nh_to_h(100.0) == pytest.approx(100.0e-9)


# ---------------------------------------------------------------------------
# Capacitance conversions
# ---------------------------------------------------------------------------

class TestCapacitanceConversions:
    def test_pf_round_trip(self) -> None:
        assert f_to_pf(pf_to_f(12.0)) == pytest.approx(12.0)

    def test_nf_round_trip(self) -> None:
        assert f_to_nf(nf_to_f(2.2)) == pytest.approx(2.2)

    def test_pf_to_f_value(self) -> None:
        assert pf_to_f(12.0) == pytest.approx(12.0e-12)


# ---------------------------------------------------------------------------
# Voltage conversions — sqrt(2) trap tests
# ---------------------------------------------------------------------------

class TestVoltageConversions:
    def test_vrms_to_vpeak(self) -> None:
        """1 V RMS → sqrt(2) V peak."""
        assert vrms_to_vpeak(1.0) == pytest.approx(SQRT2)

    def test_vpeak_to_vrms(self) -> None:
        """sqrt(2) V peak → 1 V RMS."""
        assert vpeak_to_vrms(SQRT2) == pytest.approx(1.0)

    def test_vrms_vpeak_round_trip(self) -> None:
        assert vpeak_to_vrms(vrms_to_vpeak(3.5)) == pytest.approx(3.5)

    def test_vpp_to_vrms(self) -> None:
        """1 Vpp → Vpeak = 0.5 → Vrms = 0.5/sqrt(2) = 1/(2*sqrt(2))."""
        assert vpp_to_vrms(1.0) == pytest.approx(1.0 / (2.0 * SQRT2))

    def test_vrms_to_vpp(self) -> None:
        """1 V RMS → Vpp = 2*sqrt(2)."""
        assert vrms_to_vpp(1.0) == pytest.approx(2.0 * SQRT2)

    def test_vpp_round_trip(self) -> None:
        assert vpp_to_vrms(vrms_to_vpp(5.0)) == pytest.approx(5.0)

    @given(st.floats(min_value=1e-6, max_value=1e6))
    def test_vpeak_vrms_hypothesis(self, v: float) -> None:
        """Round-trip peak ↔ RMS must be exact."""
        assert vpeak_to_vrms(vrms_to_vpeak(v)) == pytest.approx(v, rel=1e-12)


# ---------------------------------------------------------------------------
# Current conversions
# ---------------------------------------------------------------------------

class TestCurrentConversions:
    def test_irms_to_ipeak(self) -> None:
        assert irms_to_ipeak(1.0) == pytest.approx(SQRT2)

    def test_ipeak_to_irms(self) -> None:
        assert ipeak_to_irms(SQRT2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Power conversions
# ---------------------------------------------------------------------------

class TestPowerConversions:
    def test_0_dbm_is_1_mw(self) -> None:
        assert dbm_to_w(0.0) == pytest.approx(1.0e-3)

    def test_30_dbm_is_1_w(self) -> None:
        assert dbm_to_w(30.0) == pytest.approx(1.0)

    def test_20_dbm(self) -> None:
        assert dbm_to_w(20.0) == pytest.approx(0.1)

    def test_w_to_dbm_round_trip(self) -> None:
        assert w_to_dbm(dbm_to_w(13.0)) == pytest.approx(13.0)

    def test_w_to_dbm_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            w_to_dbm(-1.0)

    def test_w_to_dbm_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            w_to_dbm(0.0)


# ---------------------------------------------------------------------------
# Angle conversions
# ---------------------------------------------------------------------------

class TestAngleConversions:
    def test_deg_to_rad_90(self) -> None:
        assert deg_to_rad(90.0) == pytest.approx(math.pi / 2.0)

    def test_rad_to_deg_pi(self) -> None:
        assert rad_to_deg(math.pi) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Source convention conversions — CRITICAL factor-of-two tests
# ---------------------------------------------------------------------------

class TestSourceConventions:
    def test_available_power_to_vth_known(self) -> None:
        """P_av = 0.1 W into 50 Ω → V_th = 2*sqrt(0.1*50) = 2*sqrt(5).

        The matched-load voltage would be V_th/2 = sqrt(5) ≈ 2.236 V.
        The formula V_th = 2*sqrt(P_av*R_s) must NOT be sqrt(P_av*R_s).
        """
        v_th = available_power_to_vth_rms(0.1, 50.0)
        expected = 2.0 * math.sqrt(0.1 * 50.0)
        assert v_th == pytest.approx(expected)
        # Sanity: matched load gets half of V_th
        v_load = v_th / 2.0
        p_load = v_load**2 / 50.0
        assert p_load == pytest.approx(0.1)

    def test_20_dbm_50_ohm(self) -> None:
        """20 dBm = 100 mW.  V_th = 2*sqrt(0.1*50) ≈ 4.472 V RMS."""
        p_w = dbm_to_w(20.0)
        v_th = available_power_to_vth_rms(p_w, 50.0)
        assert v_th == pytest.approx(2.0 * math.sqrt(p_w * 50.0))

    def test_round_trip_vth_pav(self) -> None:
        """V_th → P_av → V_th must be identity."""
        v_th = 7.07
        r_s = 50.0
        p_av = vth_rms_to_available_power(v_th, r_s)
        v_th_recovered = available_power_to_vth_rms(p_av, r_s)
        assert v_th_recovered == pytest.approx(v_th)

    def test_available_power_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            available_power_to_vth_rms(-1.0, 50.0)

    def test_available_power_zero_rs_raises(self) -> None:
        with pytest.raises(ValueError):
            available_power_to_vth_rms(0.1, 0.0)

    def test_generator_rms_into_z0(self) -> None:
        """Generator shows 1 V RMS into 50 Ω → V_th = 2 V RMS."""
        v_th = generator_display_to_vth_rms(1.0, "rms_into_z0")
        assert v_th == pytest.approx(2.0)

    def test_generator_vpp_into_z0(self) -> None:
        """Generator shows 2 Vpp into 50 Ω → V_load_rms = 2/(2*sqrt(2)),
        V_th = 2 * V_load_rms."""
        v_th = generator_display_to_vth_rms(2.0, "vpp_into_z0")
        v_load_rms = 2.0 / (2.0 * SQRT2)
        assert v_th == pytest.approx(2.0 * v_load_rms)

    def test_generator_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown display convention"):
            generator_display_to_vth_rms(1.0, "bad_convention")

    @given(st.floats(min_value=1e-6, max_value=1e3))
    def test_pav_vth_round_trip_hypothesis(self, p_av: float) -> None:
        """Property: round-trip P_av → V_th → P_av is identity."""
        r_s = 50.0
        v_th = available_power_to_vth_rms(p_av, r_s)
        p_recovered = vth_rms_to_available_power(v_th, r_s)
        assert p_recovered == pytest.approx(p_av, rel=1e-10)


# ---------------------------------------------------------------------------
# Impedance / reflection
# ---------------------------------------------------------------------------

class TestImpedanceReflection:
    def test_50_ohm_perfect_match(self) -> None:
        gamma = z_to_gamma(50.0 + 0j, 50.0)
        assert abs(gamma) == pytest.approx(0.0, abs=1e-15)

    def test_open_circuit(self) -> None:
        gamma = z_to_gamma(1e12 + 0j, 50.0)
        assert abs(gamma) == pytest.approx(1.0, abs=1e-6)

    def test_short_circuit(self) -> None:
        gamma = z_to_gamma(0.0 + 0j, 50.0)
        assert gamma == pytest.approx(-1.0)

    def test_known_gamma(self) -> None:
        """Z=100 Ω, Z_ref=50 → Gamma = (100-50)/(100+50) = 1/3."""
        gamma = z_to_gamma(100.0 + 0j, 50.0)
        assert gamma.real == pytest.approx(1.0 / 3.0)
        assert gamma.imag == pytest.approx(0.0)

    def test_gamma_to_z_round_trip(self) -> None:
        z_in = 75.0 + 25.0j
        gamma = z_to_gamma(z_in, 50.0)
        z_recovered = gamma_to_z(gamma, 50.0)
        assert z_recovered.real == pytest.approx(z_in.real, rel=1e-12)
        assert z_recovered.imag == pytest.approx(z_in.imag, rel=1e-12)

    def test_s11_db_perfect_match(self) -> None:
        result = s11_db(50.0 + 0j, 50.0)
        assert result == float("-inf")

    def test_s11_db_from_gamma_perfect(self) -> None:
        result = s11_db_from_gamma(0.0 + 0j)
        assert result == float("-inf")

    def test_s11_db_known(self) -> None:
        """Z=100, Z_ref=50 → |Γ|=1/3 → S11 ≈ -9.54 dB."""
        result = s11_db(100.0 + 0j, 50.0)
        expected = 20.0 * math.log10(1.0 / 3.0)
        assert result == pytest.approx(expected)

    def test_non_50_ohm_reference(self) -> None:
        """Z=75, Z_ref=75 → perfect match."""
        gamma = z_to_gamma(75.0 + 0j, 75.0)
        assert abs(gamma) == pytest.approx(0.0, abs=1e-15)

    def test_z_to_gamma_array(self) -> None:
        z = np.array([50.0 + 0j, 100.0 + 0j, 25.0 + 0j])
        gamma = z_to_gamma_array(z, 50.0)
        assert abs(gamma[0]) == pytest.approx(0.0, abs=1e-15)
        assert gamma[1].real == pytest.approx(1.0 / 3.0)
        assert gamma[2].real == pytest.approx(-1.0 / 3.0)

    def test_s11_db_array(self) -> None:
        z = np.array([100.0 + 0j])
        result = s11_db_array(z, 50.0)
        expected = 20.0 * math.log10(1.0 / 3.0)
        assert result[0] == pytest.approx(expected)

    def test_negative_zref_raises(self) -> None:
        with pytest.raises(ValueError):
            z_to_gamma(50.0, -10.0)

    @given(
        st.floats(min_value=1.0, max_value=5000.0),
        st.floats(min_value=-5000.0, max_value=5000.0),
    )
    def test_gamma_z_round_trip_hypothesis(self, r: float, x: float) -> None:
        """Property: Z → Γ → Z must be identity for reasonable Z/Z_ref ratios.

        Note: the Γ→Z map is inherently ill-conditioned when |Γ|→1, so we
        restrict |Z| to avoid ratios beyond ~100:1 vs Z_ref.
        """
        z_in = complex(r, x)
        z_ref = 50.0
        gamma = z_to_gamma(z_in, z_ref)
        assume(abs(1.0 - gamma) > 1e-4)
        z_back = gamma_to_z(gamma, z_ref)
        assert z_back.real == pytest.approx(z_in.real, rel=1e-6)
        assert z_back.imag == pytest.approx(z_in.imag, rel=1e-6, abs=1e-6)
