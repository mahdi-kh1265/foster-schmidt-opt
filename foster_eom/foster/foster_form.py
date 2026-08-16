"""Foster-form reactance mathematics (Prompt 04A).

Implements the Foster-I canonical reactance function, its derivative,
coefficient-to-component conversion, coefficient bounds from physical
component limits, and required-pole-interval identification.

All public APIs accept and return frequencies in Hz.  The internal
variables ω = 2π·f and q_m = (2π·f_p)² are derived at the mathematical
boundary and never exposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.foster.schmidt import FosterBranchTolerances

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_DEFAULT_BRANCH_TOL = FosterBranchTolerances()


@dataclass(frozen=True)
class RequiredPoleIntervalHz:
    """A frequency interval (Hz) where at least one Foster pole is required.

    Arises from adjacent targets with X_{i+1} ≤ X_i (positive-residue
    Foster monotonicity rule).
    """

    f_lo_hz: float
    f_hi_hz: float


@dataclass(frozen=True)
class CoefficientBounds:
    """Physical coefficient bounds derived from component limits.

    k0_bounds : (k0_min, k0_max) or None if endpoint capacitor disabled.
    kinf_bounds : (kinf_min, kinf_max) or None if endpoint inductor disabled.
    km_bounds : per-cell (k_m_min, k_m_max).
    """

    k0_bounds: tuple[float, float] | None
    kinf_bounds: tuple[float, float] | None
    km_bounds: tuple[tuple[float, float], ...]
    any_infeasible: bool
    infeasible_cells: tuple[int, ...]


@dataclass(frozen=True)
class FosterCell:
    """A finite Foster LC cell."""

    l_h: float
    c_f: float
    f_pole_hz: float


@dataclass(frozen=True)
class FosterComponents:
    """Physical L/C components from Foster decomposition."""

    c0_f: float | None
    l_inf_h: float | None
    cells: tuple[FosterCell, ...]


# ---------------------------------------------------------------------------
# Internal ω/q helpers
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi


def _hz_to_omega(f_hz: np.ndarray) -> np.ndarray:
    return _TWO_PI * f_hz


def _hz_to_q(f_hz: np.ndarray) -> np.ndarray:
    omega = _TWO_PI * f_hz
    return omega * omega  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Foster reactance — internal ω/q functions (private)
# ---------------------------------------------------------------------------


def _foster_reactance_omega(
    omega: float | np.ndarray,
    k0: float,
    k_inf: float,
    k_m: np.ndarray,
    q_m: np.ndarray,
) -> np.ndarray:
    """Evaluate X(w) = -k0/w + w*k_inf + sum w*k_m/(q_m - w^2).

    Works with scalar or array w.
    """
    omega = np.asarray(omega, dtype=np.float64)
    result = np.zeros_like(omega)
    if k0 != 0.0:
        result -= k0 / omega
    if k_inf != 0.0:
        result += omega * k_inf
    for km_j, qm_j in zip(k_m, q_m, strict=True):
        result += omega * km_j / (qm_j - omega * omega)
    return result


def _foster_derivative_omega(
    omega: float | np.ndarray,
    k0: float,
    k_inf: float,
    k_m: np.ndarray,
    q_m: np.ndarray,
) -> np.ndarray:
    """Evaluate dX/dw.

    dX/dw = k0/w^2 + k_inf + sum k_m*(q_m + w^2)/(q_m - w^2)^2
    """
    omega = np.asarray(omega, dtype=np.float64)
    result = np.zeros_like(omega)
    if k0 != 0.0:
        result += k0 / (omega * omega)
    if k_inf != 0.0:
        result += k_inf
    for km_j, qm_j in zip(k_m, q_m, strict=True):
        o2 = omega * omega
        denom = qm_j - o2
        result += km_j * (qm_j + o2) / (denom * denom)
    return result


# ---------------------------------------------------------------------------
# Public Hz-based APIs
# ---------------------------------------------------------------------------


def foster_reactance_hz(
    f_hz: float | np.ndarray,
    k0: float | None,
    k_inf: float | None,
    k_m: np.ndarray,
    f_poles_hz: np.ndarray,
) -> np.ndarray:
    """Evaluate the Foster-I reactance at given frequencies.

    Public API uses Hz throughout.  Disabled endpoints (None) are treated
    as zero contribution — consistent with "absent term."

    Parameters
    ----------
    f_hz : float or ndarray
        Evaluation frequencies in Hz.
    k0 : float or None
        Endpoint capacitor coefficient.  None = disabled (no C₀ term).
    k_inf : float or None
        Endpoint inductor coefficient.  None = disabled (no L∞ term).
    k_m : ndarray
        Finite-cell residue coefficients, shape (M,).
    f_poles_hz : ndarray
        Finite-cell pole frequencies in Hz, shape (M,).

    Returns
    -------
    np.ndarray
        Reactance values in Ω.
    """
    omega = _hz_to_omega(np.asarray(f_hz, dtype=np.float64))
    q_m = _hz_to_q(np.asarray(f_poles_hz, dtype=np.float64))
    k_m = np.asarray(k_m, dtype=np.float64)
    return _foster_reactance_omega(
        omega, 0.0 if k0 is None else k0, 0.0 if k_inf is None else k_inf, k_m, q_m
    )


def foster_derivative_hz(
    f_hz: float | np.ndarray,
    k0: float | None,
    k_inf: float | None,
    k_m: np.ndarray,
    f_poles_hz: np.ndarray,
) -> np.ndarray:
    """Evaluate dX/df (derivative w.r.t. frequency in Hz).

    Returns dX/df = 2π · dX/dω, including the chain-rule factor.

    Parameters
    ----------
    f_hz, k0, k_inf, k_m, f_poles_hz
        Same as ``foster_reactance_hz``.

    Returns
    -------
    np.ndarray
        dX/df in Ω/Hz.
    """
    omega = _hz_to_omega(np.asarray(f_hz, dtype=np.float64))
    q_m = _hz_to_q(np.asarray(f_poles_hz, dtype=np.float64))
    k_m = np.asarray(k_m, dtype=np.float64)
    dxdw = _foster_derivative_omega(
        omega, 0.0 if k0 is None else k0, 0.0 if k_inf is None else k_inf, k_m, q_m
    )
    # dX/df = dX/dω · dω/df = dX/dω · 2π
    return dxdw * _TWO_PI


# ---------------------------------------------------------------------------
# Component conversion
# ---------------------------------------------------------------------------


def coefficients_to_components(
    k0: float | None,
    k_inf: float | None,
    k_m: np.ndarray,
    f_poles_hz: np.ndarray,
) -> FosterComponents:
    """Convert Foster coefficients to physical L/C components.

    Accepts f_poles_hz (Hz), not q_m.
    """
    k_m = np.asarray(k_m, dtype=np.float64)
    f_poles_hz = np.asarray(f_poles_hz, dtype=np.float64)
    q_m = _hz_to_q(f_poles_hz)

    c0_f = 1.0 / k0 if k0 is not None and k0 > 0.0 else None
    l_inf_h = k_inf if k_inf is not None and k_inf > 0.0 else None

    cells: list[FosterCell] = []
    for i in range(len(k_m)):
        km_i = float(k_m[i])
        qm_i = float(q_m[i])
        fp_i = float(f_poles_hz[i])
        c_f = 1.0 / km_i if km_i > 0.0 else math.inf
        l_h = km_i / qm_i if qm_i > 0.0 else 0.0
        cells.append(FosterCell(l_h=l_h, c_f=c_f, f_pole_hz=fp_i))

    return FosterComponents(c0_f=c0_f, l_inf_h=l_inf_h, cells=tuple(cells))


# ---------------------------------------------------------------------------
# Coefficient bounds
# ---------------------------------------------------------------------------


def compute_coefficient_bounds(
    f_poles_hz: np.ndarray,
    enable_k0: bool,
    enable_kinf: bool,
    component_limits: ContinuousLimits,
) -> CoefficientBounds:
    """Derive physical [k_min, k_max] per coefficient.

    Accepts pole frequencies in Hz.  Derives q_m internally.
    """
    f_poles_hz = np.asarray(f_poles_hz, dtype=np.float64)
    q_m = _hz_to_q(f_poles_hz)

    c_min = component_limits.c_min_f
    c_max = component_limits.c_max_f
    l_min = component_limits.l_min_h
    l_max = component_limits.l_max_h

    k0_bounds: tuple[float, float] | None = None
    if enable_k0:
        k0_bounds = (1.0 / c_max, 1.0 / c_min)

    kinf_bounds: tuple[float, float] | None = None
    if enable_kinf:
        kinf_bounds = (l_min, l_max)

    km_bounds_list: list[tuple[float, float]] = []
    infeasible_cells: list[int] = []
    for idx, qm_j in enumerate(q_m):
        qm_val = float(qm_j)
        km_min = max(1.0 / c_max, qm_val * l_min)
        km_max = min(1.0 / c_min, qm_val * l_max)
        km_bounds_list.append((km_min, km_max))
        if km_min > km_max:
            infeasible_cells.append(idx)

    return CoefficientBounds(
        k0_bounds=k0_bounds,
        kinf_bounds=kinf_bounds,
        km_bounds=tuple(km_bounds_list),
        any_infeasible=len(infeasible_cells) > 0,
        infeasible_cells=tuple(infeasible_cells),
    )


# ---------------------------------------------------------------------------
# Required-pole intervals
# ---------------------------------------------------------------------------


def find_required_pole_intervals(
    f_targets_hz: np.ndarray,
    x_targets: np.ndarray,
    r_match_ohm: float = 50.0,
    branch_tolerances: FosterBranchTolerances | None = None,
) -> list[RequiredPoleIntervalHz]:
    """Identify Hz intervals where X decreases, requiring ≥ 1 pole.

    Precondition: branch has been classified as FINITE_FOSTER.
    If all targets are within zero tolerance, returns empty list.

    Parameters
    ----------
    f_targets_hz : ndarray
        Target frequencies in Hz, shape (N,), must be sorted ascending.
    x_targets : ndarray
        Target reactance values in Ω, shape (N,).
    r_match_ohm : float
        Match resistance for zero-target scale reference.
    branch_tolerances : FosterBranchTolerances | None
        Zero-target tolerances.

    Returns
    -------
    list[RequiredPoleIntervalHz]
    """
    f_targets_hz = np.asarray(f_targets_hz, dtype=np.float64).ravel()
    x_targets = np.asarray(x_targets, dtype=np.float64).ravel()
    if len(f_targets_hz) != len(x_targets):
        raise ValueError("f_targets_hz and x_targets must have the same length")
    if len(f_targets_hz) < 2:
        return []

    bt = branch_tolerances or _DEFAULT_BRANCH_TOL

    # Check if all-zero (should have been routed to ZERO_IMPEDANCE)
    zero_thresh = bt.x_zero_abs + bt.x_zero_rel * r_match_ohm
    if all(abs(x) <= zero_thresh for x in x_targets):
        return []  # Trivial zero: no poles needed

    intervals: list[RequiredPoleIntervalHz] = []
    for i in range(len(f_targets_hz) - 1):
        if x_targets[i + 1] <= x_targets[i]:
            intervals.append(
                RequiredPoleIntervalHz(
                    f_lo_hz=float(f_targets_hz[i]),
                    f_hi_hz=float(f_targets_hz[i + 1]),
                )
            )
    return intervals
