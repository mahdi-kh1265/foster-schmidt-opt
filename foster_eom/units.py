"""SI unit conversion utilities.

All internal numerical code uses SI base units (Hz, H, F, ohm, V RMS, A RMS,
W, rad).  Conversion happens only at schema/import/export/GUI boundaries.

Source-convention conversions (Thévenin ↔ available power) are included here
because they are pure unit/convention transforms, not circuit physics.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------
HZ_PER_KHZ: Final[float] = 1.0e3
HZ_PER_MHZ: Final[float] = 1.0e6
HZ_PER_GHZ: Final[float] = 1.0e9


def khz_to_hz(f_khz: float) -> float:
    """Convert kHz → Hz."""
    return f_khz * HZ_PER_KHZ


def mhz_to_hz(f_mhz: float) -> float:
    """Convert MHz → Hz."""
    return f_mhz * HZ_PER_MHZ


def ghz_to_hz(f_ghz: float) -> float:
    """Convert GHz → Hz."""
    return f_ghz * HZ_PER_GHZ


def hz_to_khz(f_hz: float) -> float:
    """Convert Hz → kHz."""
    return f_hz / HZ_PER_KHZ


def hz_to_mhz(f_hz: float) -> float:
    """Convert Hz → MHz."""
    return f_hz / HZ_PER_MHZ


def hz_to_ghz(f_hz: float) -> float:
    """Convert Hz → GHz."""
    return f_hz / HZ_PER_GHZ


def hz_to_rad_per_s(f_hz: float) -> float:
    """Convert frequency in Hz to angular frequency in rad/s."""
    return 2.0 * math.pi * f_hz


def rad_per_s_to_hz(omega: float) -> float:
    """Convert angular frequency in rad/s to frequency in Hz."""
    return omega / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Inductance
# ---------------------------------------------------------------------------
H_PER_UH: Final[float] = 1.0e-6
H_PER_NH: Final[float] = 1.0e-9


def uh_to_h(l_uh: float) -> float:
    """Convert µH → H."""
    return l_uh * H_PER_UH


def nh_to_h(l_nh: float) -> float:
    """Convert nH → H."""
    return l_nh * H_PER_NH


def h_to_uh(l_h: float) -> float:
    """Convert H → µH."""
    return l_h / H_PER_UH


def h_to_nh(l_h: float) -> float:
    """Convert H → nH."""
    return l_h / H_PER_NH


# ---------------------------------------------------------------------------
# Capacitance
# ---------------------------------------------------------------------------
F_PER_NF: Final[float] = 1.0e-9
F_PER_PF: Final[float] = 1.0e-12


def nf_to_f(c_nf: float) -> float:
    """Convert nF → F."""
    return c_nf * F_PER_NF


def pf_to_f(c_pf: float) -> float:
    """Convert pF → F."""
    return c_pf * F_PER_PF


def f_to_nf(c_f: float) -> float:
    """Convert F → nF."""
    return c_f / F_PER_NF


def f_to_pf(c_f: float) -> float:
    """Convert F → pF."""
    return c_f / F_PER_PF


# ---------------------------------------------------------------------------
# Voltage  (RMS is the internal convention)
# ---------------------------------------------------------------------------
SQRT2: Final[float] = math.sqrt(2.0)


def vpeak_to_vrms(v_peak: float) -> float:
    """Convert sinusoidal peak voltage to RMS voltage."""
    return v_peak / SQRT2


def vrms_to_vpeak(v_rms: float) -> float:
    """Convert sinusoidal RMS voltage to peak voltage."""
    return v_rms * SQRT2


def vpp_to_vrms(v_pp: float) -> float:
    """Convert peak-to-peak voltage to RMS voltage (sinusoidal assumption)."""
    return v_pp / (2.0 * SQRT2)


def vrms_to_vpp(v_rms: float) -> float:
    """Convert RMS voltage to peak-to-peak voltage (sinusoidal assumption)."""
    return v_rms * 2.0 * SQRT2


# ---------------------------------------------------------------------------
# Current  (RMS is the internal convention)
# ---------------------------------------------------------------------------

def ipeak_to_irms(i_peak: float) -> float:
    """Convert sinusoidal peak current to RMS current."""
    return i_peak / SQRT2


def irms_to_ipeak(i_rms: float) -> float:
    """Convert sinusoidal RMS current to peak current."""
    return i_rms * SQRT2


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def dbm_to_w(p_dbm: float) -> float:
    """Convert dBm to watts."""
    return 1.0e-3 * 10.0 ** (p_dbm / 10.0)


def w_to_dbm(p_w: float) -> float:
    """Convert watts to dBm.  Requires p_w > 0."""
    if p_w <= 0.0:
        raise ValueError(f"Power must be positive for dBm conversion, got {p_w}")
    return 10.0 * math.log10(p_w / 1.0e-3)


# ---------------------------------------------------------------------------
# Angle
# ---------------------------------------------------------------------------

def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(deg)


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(rad)


# ---------------------------------------------------------------------------
# Source-convention conversions  (spec §3.2)
# ---------------------------------------------------------------------------

def available_power_to_vth_rms(p_av_w: float, r_s: float) -> float:
    """Compute Thévenin RMS voltage from available power and real source resistance.

    For a real source resistance R_s and available power P_av (power delivered
    to a conjugate-matched load):

        V_th,rms = 2 * sqrt(P_av * R_s)

    This is NOT V_th = sqrt(P_av * R_s) — the factor of 2 accounts for the
    voltage divider at matched load (V_load = V_th/2 at conjugate match).

    Parameters
    ----------
    p_av_w : float
        Available power in watts (must be ≥ 0).
    r_s : float
        Real source resistance in ohms (must be > 0).

    Returns
    -------
    float
        Thévenin RMS open-circuit voltage in volts.

    Raises
    ------
    ValueError
        If p_av_w < 0 or r_s <= 0.
    """
    if p_av_w < 0.0:
        raise ValueError(f"Available power must be non-negative, got {p_av_w}")
    if r_s <= 0.0:
        raise ValueError(f"Source resistance must be positive, got {r_s}")
    return 2.0 * math.sqrt(p_av_w * r_s)


def vth_rms_to_available_power(v_th_rms: float, r_s: float) -> float:
    """Compute available power from Thévenin RMS voltage and real source resistance.

    P_av = V_th_rms^2 / (4 * R_s)

    Parameters
    ----------
    v_th_rms : float
        Thévenin RMS open-circuit voltage in volts.
    r_s : float
        Real source resistance in ohms (must be > 0).

    Returns
    -------
    float
        Available power in watts.
    """
    if r_s <= 0.0:
        raise ValueError(f"Source resistance must be positive, got {r_s}")
    return v_th_rms**2 / (4.0 * r_s)


def generator_display_to_vth_rms(
    v_display: float,
    display_convention: str,
    r_s: float = 50.0,
) -> float:
    """Convert a generator "display" voltage to Thévenin RMS voltage.

    Many RF generators display "voltage into 50 Ω" which is V_load = V_th/2
    at matched load.  This function handles the various labeling conventions.

    Parameters
    ----------
    v_display : float
        Voltage value shown on the generator display.
    display_convention : str
        One of ``"rms_into_z0"``, ``"vpp_into_z0"``, ``"peak_into_z0"``.
    r_s : float
        Source resistance (default 50 Ω).

    Returns
    -------
    float
        Thévenin RMS open-circuit voltage.
    """
    if display_convention == "rms_into_z0":
        # Display shows RMS voltage across matched load: V_load = V_th/2
        return 2.0 * v_display
    elif display_convention == "vpp_into_z0":
        # Display shows Vpp across matched load
        v_load_rms = vpp_to_vrms(v_display)
        return 2.0 * v_load_rms
    elif display_convention == "peak_into_z0":
        # Display shows peak voltage across matched load
        v_load_rms = vpeak_to_vrms(v_display)
        return 2.0 * v_load_rms
    else:
        raise ValueError(
            f"Unknown display convention '{display_convention}'. "
            f"Expected one of: rms_into_z0, vpp_into_z0, peak_into_z0"
        )


# ---------------------------------------------------------------------------
# Impedance / reflection  (spec §3.4)
# ---------------------------------------------------------------------------

def z_to_gamma(z_in: complex, z_ref: float = 50.0) -> complex:
    """Compute reflection coefficient from impedance.

    Gamma = (Z_in - Z_ref) / (Z_in + Z_ref)

    Parameters
    ----------
    z_in : complex
        Input impedance.
    z_ref : float
        Real positive reference impedance (default 50 Ω).

    Returns
    -------
    complex
        Reflection coefficient.
    """
    if z_ref <= 0.0:
        raise ValueError(f"Reference impedance must be positive, got {z_ref}")
    return (z_in - z_ref) / (z_in + z_ref)


def gamma_to_z(gamma: complex, z_ref: float = 50.0) -> complex:
    """Compute impedance from reflection coefficient.

    Z_in = Z_ref * (1 + Gamma) / (1 - Gamma)

    Parameters
    ----------
    gamma : complex
        Reflection coefficient.
    z_ref : float
        Real positive reference impedance (default 50 Ω).

    Returns
    -------
    complex
        Input impedance.
    """
    if z_ref <= 0.0:
        raise ValueError(f"Reference impedance must be positive, got {z_ref}")
    return z_ref * (1.0 + gamma) / (1.0 - gamma)


def s11_db(z_in: complex, z_ref: float = 50.0) -> float:
    """Compute S11 in dB from impedance.

    S11_dB = 20 * log10(|Gamma|)

    Parameters
    ----------
    z_in : complex
        Input impedance.
    z_ref : float
        Reference impedance.

    Returns
    -------
    float
        S11 in dB.  Returns -inf for perfect match.
    """
    gamma = z_to_gamma(z_in, z_ref)
    mag = abs(gamma)
    if mag == 0.0:
        return float("-inf")
    return 20.0 * math.log10(mag)


def s11_db_from_gamma(gamma: complex) -> float:
    """Compute S11 in dB from reflection coefficient magnitude."""
    mag = abs(gamma)
    if mag == 0.0:
        return float("-inf")
    return 20.0 * math.log10(mag)


# ---------------------------------------------------------------------------
# Vectorized helpers (NumPy)
# ---------------------------------------------------------------------------

def z_to_gamma_array(
    z_in: np.ndarray, z_ref: float = 50.0
) -> np.ndarray:
    """Vectorized reflection coefficient computation."""
    return (z_in - z_ref) / (z_in + z_ref)


def s11_db_array(z_in: np.ndarray, z_ref: float = 50.0) -> np.ndarray:
    """Vectorized S11 in dB."""
    gamma = z_to_gamma_array(z_in, z_ref)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(gamma))
