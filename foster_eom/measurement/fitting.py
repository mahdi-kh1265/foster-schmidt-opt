"""Equivalent-circuit fitting to measured data (Prompt 07).

Fits ``LossyCapacitorEOM`` or ``MBVDModel`` to a ``MeasuredDataset`` using
nonlinear least-squares with log-parameterization for positive quantities,
multi-domain residuals (S11/Z/Y), SVD-based covariance, and joint multi-start
optimization for multi-branch mBVD.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from foster_eom.measurement.dataset import (
    MeasuredDataset,
    SourceQuantity,
    _z_to_s11,
    _z_to_y,
)
from foster_eom.models.base import OnePortModel

# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------


class FitDomain(enum.StrEnum):
    """Domain for residual computation."""

    S11 = "S11"
    Z = "Z"
    Y = "Y"


@dataclass(frozen=True)
class FitDiagnostics:
    """Fit quality diagnostics.

    Attributes
    ----------
    residuals_complex : np.ndarray
        Per-frequency complex residuals in fit_domain units.
    rms_error : float
        RMS |residual| in fit_domain units.
    max_error : float
        Max |residual| in fit_domain units.
    rms_error_ohm : float
        RMS |Z_fit - Z_meas| always in Ω.
    max_error_ohm : float
        Max |Z_fit - Z_meas| always in Ω.
    fit_domain : FitDomain
        Which domain the fitting was performed in.
    converged : bool
        Whether the optimizer converged.
    message : str
        Optimizer status message.
    n_function_evals : int
        Number of function evaluations.
    jacobian_rank : int | None
        Rank of the Jacobian at the solution.
    condition_number : float | None
        Condition number of the Jacobian.
    param_covariance : np.ndarray | None
        Parameter covariance matrix; None if unreliable.
    covariance_reason : str | None
        Explanation if covariance is None.
    """

    residuals_complex: np.ndarray
    rms_error: float
    max_error: float
    rms_error_ohm: float
    max_error_ohm: float
    fit_domain: FitDomain
    converged: bool
    message: str
    n_function_evals: int
    jacobian_rank: int | None
    condition_number: float | None
    param_covariance: np.ndarray | None
    covariance_reason: str | None


@dataclass(frozen=True)
class FitResult:
    """Complete result of an equivalent-circuit fit.

    Attributes
    ----------
    model_type : str
        ``"lossy_cap"`` or ``"mbvd"``.
    model : OnePortModel
        Fitted analytic model instance.
    diagnostics : FitDiagnostics
    dataset : MeasuredDataset
        Reference to original measured truth.
    fit_domain : FitDomain
    schema_version : str
        For persistence versioning.
    """

    model_type: str
    model: OnePortModel
    diagnostics: FitDiagnostics
    dataset: MeasuredDataset
    fit_domain: FitDomain
    schema_version: str = "p07.1"


# ---------------------------------------------------------------------------
# Domain evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_domain(
    z_model: np.ndarray,
    z_meas: np.ndarray,
    s11_meas: np.ndarray,
    y_meas: np.ndarray,
    z_ref: float,
    domain: FitDomain,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fit-domain residuals and weights.

    Returns (fit_values, meas_values) as complex arrays in the chosen domain.
    """
    if domain == FitDomain.S11:
        s11_fit, _ = _z_to_s11(z_model, z_ref)
        return s11_fit, s11_meas
    elif domain == FitDomain.Y:
        y_fit, _ = _z_to_y(z_model)
        return y_fit, y_meas
    else:  # Z
        return z_model, z_meas


def _build_residuals(
    fit_vals: np.ndarray,
    meas_vals: np.ndarray,
    domain: FitDomain,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Build real residual vector from complex residuals.

    For S11 domain: uniform weighting by default (no 1/|S11|).
    For Z/Y domain: normalized weighting by default.
    """
    diff = fit_vals - meas_vals

    if weights is not None:
        w = weights
    elif domain == FitDomain.S11:
        # Uniform weighting for S11 (user correction #3)
        w = np.ones(len(diff), dtype=np.float64)
    else:
        # Normalized weighting for Z/Y domains
        meas_mag = np.abs(meas_vals)
        median_mag = float(np.median(meas_mag))
        floor = max(median_mag * 0.01, 1e-15)
        w = 1.0 / np.maximum(meas_mag, floor)

    # Stack Re/Im into a real vector
    return np.concatenate(
        [
            (diff.real * w),
            (diff.imag * w),
        ]
    )


# ---------------------------------------------------------------------------
# SVD-based covariance
# ---------------------------------------------------------------------------


def _compute_covariance(
    jac: np.ndarray, residuals: np.ndarray, n_params: int
) -> tuple[np.ndarray | None, int | None, float | None, str | None]:
    """Compute parameter covariance using SVD/pseudoinverse.

    Returns (covariance, rank, condition_number, reason_if_none).
    """
    try:
        _U, s, Vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None, "SVD failed"

    rank = int(np.sum(s > s[0] * 1e-10))
    cond = float(s[0] / s[max(rank - 1, 0)]) if rank > 0 else float("inf")
    dof = len(residuals) - n_params

    if rank < n_params:
        return None, rank, cond, f"rank-deficient: rank={rank}/{n_params}"
    if cond > 1e8:
        return None, rank, cond, f"ill-conditioned: cond={cond:.2e}"
    if dof <= 0:
        return None, rank, cond, f"insufficient degrees of freedom: dof={dof}"

    residual_var = float(np.dot(residuals, residuals)) / dof
    s_inv_sq = np.diag(1.0 / s**2)
    covariance = (Vt.T @ s_inv_sq @ Vt) * residual_var

    return covariance, rank, cond, None


# ---------------------------------------------------------------------------
# Model evaluation for fitting
# ---------------------------------------------------------------------------


def _eval_lossy_cap_z(f_hz: np.ndarray, c0: float, rs: float, ls: float, g0: float) -> np.ndarray:
    """Evaluate lossy capacitor impedance without constructing a model object."""
    omega = 2.0 * np.pi * f_hz
    z_series = rs + 1j * omega * ls
    y_core = g0 + 1j * omega * c0
    with np.errstate(divide="ignore", invalid="ignore"):
        return z_series + (1.0 / y_core)


def _eval_mbvd_z(
    f_hz: np.ndarray,
    c0: float,
    g0: float,
    rs: float,
    ls: float,
    motional: list[tuple[float, float, float]],  # [(rm, lm, cm), ...]
) -> np.ndarray:
    """Evaluate mBVD impedance without constructing a model object."""
    omega = 2.0 * np.pi * f_hz
    y_core = g0 + 1j * omega * c0
    for rm, lm, cm in motional:
        with np.errstate(divide="ignore", invalid="ignore"):
            z_m = rm + 1j * omega * lm + 1.0 / (1j * omega * cm)
            z_m_arr = np.asarray(z_m, dtype=np.complex128)
            y_m = np.where(z_m_arr == 0j, np.complex128(np.inf), 1.0 / z_m_arr)
            y_core = y_core + y_m
    z_series = rs + 1j * omega * ls
    with np.errstate(divide="ignore", invalid="ignore"):
        return z_series + (1.0 / y_core)


# ---------------------------------------------------------------------------
# Log-space transform helpers
# ---------------------------------------------------------------------------


def _to_internal(params: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    """Transform physical → internal (log for strictly positive)."""
    out = params.copy()
    out[log_mask] = np.log(params[log_mask])
    return out


def _to_physical(internal: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    """Transform internal → physical (exp for log-space params)."""
    out = internal.copy()
    out[log_mask] = np.exp(internal[log_mask])
    return out


# ---------------------------------------------------------------------------
# fit_lossy_cap
# ---------------------------------------------------------------------------


def fit_lossy_cap(
    dataset: MeasuredDataset,
    *,
    domain: str = "auto",
    weights: np.ndarray | None = None,
) -> FitResult:
    """Fit a lossy capacitor model to measured data.

    Parameters
    ----------
    dataset : MeasuredDataset
    domain : str
        ``"auto"``, ``"S11"``, ``"Z"``, or ``"Y"``.
    weights : np.ndarray | None
        Per-frequency weights; uniform for S11 if None.

    Returns
    -------
    FitResult
    """
    from foster_eom.models.eom_lossy import LossyCapacitorEOM

    fit_domain = _resolve_domain(domain, dataset)
    f_hz = np.array(dataset.f_hz, copy=True)
    s11_meas = np.array(dataset.s11_complex, copy=True)
    z_meas = np.array(dataset.z_complex, copy=True)
    y_meas = np.array(dataset.y_complex, copy=True)
    z_ref = dataset.z_ref_ohm

    # Robust initialization from low-frequency capacitive region
    n_pts = len(f_hz)
    n_low = max(n_pts // 10, 2)
    f_low = f_hz[:n_low]
    z_low = z_meas[:n_low]

    # Use non-singular points for init
    valid = np.isfinite(z_low)
    if np.sum(valid) < 2:
        valid = np.isfinite(z_meas)
        f_low = f_hz[valid]
        z_low = z_meas[valid]

    omega_med = 2.0 * np.pi * float(np.median(f_low))
    im_z_med = float(np.median(z_low[np.isfinite(z_low)].imag))
    re_z_med = float(np.median(z_low[np.isfinite(z_low)].real))

    c0_init = max(-1.0 / (omega_med * im_z_med), 1e-15) if im_z_med < -1e-10 else 3.3e-12
    rs_init = max(re_z_med, 0.0)
    ls_init = 1e-12  # small positive seed
    g0_init = 1e-6  # small positive seed

    # Parameter order: [C0, Rs, Ls, G0]
    # Log-space for strictly positive: C0
    # Linear for zero-bounded: Rs, Ls, G0
    p0_phys = np.array([c0_init, rs_init, ls_init, g0_init])

    # C0 always positive (log); Rs, Ls, G0 are zero-bounded (linear)
    log_mask = np.array([True, False, False, False])

    bounds_phys = (
        np.array([1e-15, 0.0, 0.0, 0.0]),
        np.array([1e-6, 1e4, 1e-3, 1e-1]),
    )

    # Transform to internal; log(0) is avoided since only C0 uses log-space
    # and its lower bound is 1e-15 > 0
    p0_int = _to_internal(p0_phys, log_mask)
    lb_int = bounds_phys[0].copy()
    ub_int = bounds_phys[1].copy()
    lb_int[0] = np.log(bounds_phys[0][0])  # C0 lower bound in log-space
    ub_int[0] = np.log(bounds_phys[1][0])  # C0 upper bound in log-space

    # Clip initial point to be within bounds
    p0_int = np.clip(p0_int, lb_int, ub_int)

    def residual_fn(p_int: np.ndarray) -> np.ndarray:
        p_phys = _to_physical(p_int, log_mask)
        c0, rs, ls, g0 = p_phys
        z_fit = _eval_lossy_cap_z(f_hz, c0, rs, ls, g0)
        fit_vals, meas_vals = _evaluate_domain(z_fit, z_meas, s11_meas, y_meas, z_ref, fit_domain)
        return _build_residuals(fit_vals, meas_vals, fit_domain, weights)

    result = least_squares(
        residual_fn,
        p0_int,
        bounds=(lb_int, ub_int),
        method="trf",
        max_nfev=10000,
    )

    p_final = _to_physical(result.x, log_mask)
    c0, rs, ls, g0 = p_final

    model = LossyCapacitorEOM(
        c0_f=float(c0),
        rs_ohm=float(max(rs, 0.0)),
        ls_h=float(max(ls, 0.0)),
        g0_s=float(max(g0, 0.0)),
        validity_hz=dataset.validity_hz,
    )

    diagnostics = _build_diagnostics(model, dataset, fit_domain, result, len(p_final), weights)

    return FitResult(
        model_type="lossy_cap",
        model=model,
        diagnostics=diagnostics,
        dataset=dataset,
        fit_domain=fit_domain,
    )


# ---------------------------------------------------------------------------
# fit_mbvd
# ---------------------------------------------------------------------------


def fit_mbvd(
    dataset: MeasuredDataset,
    n_motional: int = 1,
    *,
    domain: str = "auto",
    weights: np.ndarray | None = None,
    n_starts: int = 3,
) -> FitResult:
    """Fit an mBVD model to measured data.

    Parameters
    ----------
    dataset : MeasuredDataset
    n_motional : int
        Number of motional branches.
    domain : str
        ``"auto"``, ``"S11"``, ``"Z"``, or ``"Y"``.
    weights : np.ndarray | None
    n_starts : int
        Number of random restarts for multi-start optimization.

    Returns
    -------
    FitResult
    """
    from foster_eom.domain.eom import MotionalBranch
    from foster_eom.models.eom_mbvd import MBVDModel

    fit_domain = _resolve_domain(domain, dataset)
    f_hz = np.array(dataset.f_hz, copy=True)
    s11_meas = np.array(dataset.s11_complex, copy=True)
    z_meas = np.array(dataset.z_complex, copy=True)
    y_meas = np.array(dataset.y_complex, copy=True)
    z_ref = dataset.z_ref_ohm

    # Step 1: Baseline from lossy cap fit (for C0, Rs, Ls init)
    lossy_result = fit_lossy_cap(dataset, domain=domain, weights=weights)
    lossy_meta = lossy_result.model.metadata()
    c0_init = lossy_meta["c0_f"]
    rs_init = lossy_meta["rs_ohm"]
    ls_init = lossy_meta["ls_h"]
    g0_init = lossy_meta.get("g0_s", 0.0)

    # Step 2: Residual peeling for motional branch initialization
    z_lossy = _eval_lossy_cap_z(f_hz, c0_init, rs_init, ls_init, g0_init)
    dz = z_meas - z_lossy

    branch_inits: list[tuple[float, float, float]] = []
    dz_remaining = dz.copy()

    for _ in range(n_motional):
        # Find peak of |ΔZ_remaining| (ignoring singular points)
        valid = np.isfinite(dz_remaining)
        if not np.any(valid):
            # Fallback: generic init
            branch_inits.append((1.0, 1e-6, 1e-12))
            continue

        mag = np.abs(dz_remaining)
        mag[~valid] = 0.0
        peak_idx = int(np.argmax(mag))
        f_res = float(f_hz[peak_idx])
        omega_res = 2.0 * np.pi * f_res

        # Estimate motional params from peak
        rm_init = max(float(np.abs(dz_remaining[peak_idx].real)), 0.01)
        # C_m and L_m from resonance frequency: f_res = 1/(2π√(Lm*Cm))
        # Use a reasonable bandwidth estimate
        cm_init = max(1.0 / (omega_res * rm_init * 10.0), 1e-18)
        lm_init = 1.0 / (omega_res**2 * cm_init)

        branch_inits.append((rm_init, lm_init, cm_init))

        # Subtract this branch's contribution for next peeling iteration
        z_branch = (
            _eval_mbvd_z(f_hz, c0_init, g0_init, rs_init, ls_init, [branch_inits[-1]]) - z_lossy
        )
        dz_remaining = dz_remaining - z_branch

    # Step 3: Joint optimization with multi-start
    # Parameter layout: [C0, G0, Rs, Ls, Rm1, Lm1, Cm1, Rm2, Lm2, Cm2, ...]
    n_base = 4
    n_branch_params = 3
    n_total = n_base + n_motional * n_branch_params

    # Log mask: C0 (log), G0 (linear), Rs (linear), Ls (linear),
    # then for each branch: Rm (linear), Lm (log), Cm (log)
    log_mask = np.zeros(n_total, dtype=bool)
    log_mask[0] = True  # C0
    for k in range(n_motional):
        offset = n_base + k * n_branch_params
        log_mask[offset + 1] = True  # Lm
        log_mask[offset + 2] = True  # Cm

    # Bounds
    lb_phys = np.zeros(n_total)
    ub_phys = np.zeros(n_total)
    # Base: C0, G0, Rs, Ls
    lb_phys[:4] = [1e-15, 0.0, 0.0, 0.0]
    ub_phys[:4] = [1e-6, 1e-1, 1e4, 1e-3]
    for k in range(n_motional):
        offset = n_base + k * n_branch_params
        lb_phys[offset : offset + 3] = [0.0, 1e-15, 1e-18]  # Rm, Lm, Cm
        ub_phys[offset : offset + 3] = [1e6, 1e-1, 1e-3]

    def residual_fn(p_int: np.ndarray) -> np.ndarray:
        p_phys = _to_physical(p_int, log_mask)
        c0, g0, rs, ls = p_phys[:4]
        branches = []
        for k in range(n_motional):
            offset = n_base + k * n_branch_params
            rm, lm, cm = p_phys[offset : offset + 3]
            branches.append((rm, lm, cm))
        z_fit = _eval_mbvd_z(f_hz, c0, g0, rs, ls, branches)
        fit_vals, meas_vals = _evaluate_domain(z_fit, z_meas, s11_meas, y_meas, z_ref, fit_domain)
        return _build_residuals(fit_vals, meas_vals, fit_domain, weights)

    # Build initial guesses
    p0_base = np.array([c0_init, g0_init, rs_init, ls_init])
    p0_branches = np.concatenate([np.array([rm, lm, cm]) for rm, lm, cm in branch_inits])
    p0_phys = np.concatenate([p0_base, p0_branches])

    # Clip to bounds
    p0_phys = np.clip(p0_phys, lb_phys + 1e-20, ub_phys - 1e-20)

    lb_int = _to_internal(lb_phys.copy(), log_mask)
    ub_int = _to_internal(ub_phys.copy(), log_mask)

    # Multi-start: peeled init + random perturbations
    rng = np.random.default_rng(42)
    starts = [p0_phys.copy()]
    for _ in range(max(n_starts - 1, 0)):
        perturbed = p0_phys * rng.lognormal(0, 0.3, size=len(p0_phys))
        perturbed = np.clip(perturbed, lb_phys + 1e-20, ub_phys - 1e-20)
        starts.append(perturbed)

    best_result = None
    best_cost = float("inf")

    for start in starts:
        p0_int = _to_internal(start, log_mask)
        p0_int = np.clip(p0_int, lb_int, ub_int)
        try:
            result = least_squares(
                residual_fn,
                p0_int,
                bounds=(lb_int, ub_int),
                method="trf",
                max_nfev=10000,
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue  # skip failed starts

    if best_result is None:
        raise RuntimeError("All multi-start optimizations failed for mBVD fit.")

    p_final = _to_physical(best_result.x, log_mask)
    c0, g0, rs, ls = p_final[:4]

    # Extract and canonicalize branches by ascending resonance frequency
    raw_branches: list[tuple[float, float, float, float]] = []
    for k in range(n_motional):
        offset = n_base + k * n_branch_params
        rm, lm, cm = p_final[offset : offset + 3]
        f0_k = 1.0 / (2.0 * math.pi * math.sqrt(max(lm * cm, 1e-30)))
        raw_branches.append((f0_k, float(max(rm, 0.0)), float(lm), float(cm)))

    raw_branches.sort(key=lambda x: x[0])  # sort by f0

    motional_branches = [
        MotionalBranch(rm_ohm=rm, lm_h=lm, cm_f=cm) for _, rm, lm, cm in raw_branches
    ]

    model = MBVDModel(
        c0_f=float(c0),
        g0_s=float(max(g0, 0.0)),
        rs_ohm=float(max(rs, 0.0)),
        ls_h=float(max(ls, 0.0)),
        motional_branches=motional_branches,
        validity_hz=dataset.validity_hz,
    )

    diagnostics = _build_diagnostics(model, dataset, fit_domain, best_result, n_total, weights)

    return FitResult(
        model_type="mbvd",
        model=model,
        diagnostics=diagnostics,
        dataset=dataset,
        fit_domain=fit_domain,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_domain(domain: str, dataset: MeasuredDataset) -> FitDomain:
    """Resolve 'auto' domain to a concrete FitDomain."""
    d = domain.upper()
    if d == "AUTO":
        if dataset.source_quantity == SourceQuantity.S11:
            return FitDomain.S11
        else:
            return FitDomain.Z
    try:
        return FitDomain(d)
    except ValueError:
        raise ValueError(
            f"Unknown fit domain '{domain}'. Valid: 'auto', 'S11', 'Z', 'Y'."
        ) from None


def _build_diagnostics(
    model: OnePortModel,
    dataset: MeasuredDataset,
    fit_domain: FitDomain,
    scipy_result: Any,
    n_params: int,
    weights: np.ndarray | None,
) -> FitDiagnostics:
    """Build FitDiagnostics from a scipy least_squares result."""
    f_hz = np.array(dataset.f_hz, copy=True)
    z_meas = np.array(dataset.z_complex, copy=True)
    s11_meas = np.array(dataset.s11_complex, copy=True)
    y_meas = np.array(dataset.y_complex, copy=True)
    z_ref = dataset.z_ref_ohm

    z_fit = np.asarray(model.z(f_hz), dtype=np.complex128)

    # Domain residuals
    fit_vals, meas_vals = _evaluate_domain(z_fit, z_meas, s11_meas, y_meas, z_ref, fit_domain)
    domain_resid = fit_vals - meas_vals

    # Z residuals (always)
    z_resid = z_fit - z_meas
    z_err_mag = np.abs(z_resid)
    # Exclude singular / infinite points
    finite_mask = np.isfinite(z_err_mag)
    if np.any(finite_mask):
        rms_z = float(np.sqrt(np.mean(z_err_mag[finite_mask] ** 2)))
        max_z = float(np.max(z_err_mag[finite_mask]))
    else:
        rms_z = float("inf")
        max_z = float("inf")

    domain_err_mag = np.abs(domain_resid)
    finite_d = np.isfinite(domain_err_mag)
    if np.any(finite_d):
        rms_d = float(np.sqrt(np.mean(domain_err_mag[finite_d] ** 2)))
        max_d = float(np.max(domain_err_mag[finite_d]))
    else:
        rms_d = float("inf")
        max_d = float("inf")

    converged = scipy_result.status > 0

    # Covariance
    jac = scipy_result.jac
    resid_vec = scipy_result.fun
    cov, rank, cond, reason = _compute_covariance(jac, resid_vec, n_params)

    return FitDiagnostics(
        residuals_complex=domain_resid,
        rms_error=rms_d,
        max_error=max_d,
        rms_error_ohm=rms_z,
        max_error_ohm=max_z,
        fit_domain=fit_domain,
        converged=converged,
        message=scipy_result.message,
        n_function_evals=scipy_result.nfev,
        jacobian_rank=rank,
        condition_number=cond,
        param_covariance=cov,
        covariance_reason=reason,
    )
