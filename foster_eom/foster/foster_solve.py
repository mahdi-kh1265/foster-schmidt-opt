"""Scaled constrained Foster coefficient solve (Prompt 04A).

Builds a self-describing linear system with per-column metadata,
geometric-mean column scaling, and row scaling.  Solves via bounded
least squares with a two-stage underdetermined strategy (minimum
bounded residual → regularized selection with degradation guard).

All public APIs accept frequencies in Hz.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear  # type: ignore[import-untyped]

from foster_eom.foster.foster_form import (
    _TWO_PI,
    CoefficientBounds,
)

# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------


class CoefficientKind(enum.StrEnum):
    """Kind of Foster coefficient column."""

    K0 = "k0"  # endpoint capacitor: C₀ = 1/k₀
    K_INF = "k_inf"  # endpoint inductor: L∞ = k∞
    K_RESIDUE = "k_m"  # finite Foster cell


@dataclass(frozen=True)
class FosterCoefficientDescriptor:
    """Metadata for one column of the Foster design matrix."""

    kind: CoefficientKind
    cell_index: int | None  # None for K0, K_INF
    scale: float  # s_j for normalization
    lower_bound: float  # physical k_j,min
    upper_bound: float  # physical k_j,max


# ---------------------------------------------------------------------------
# Linear system
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FosterLinearSystem:
    """Self-describing, pre-scaled Foster interpolation linear system.

    Single source of truth for column identity, physical bounds,
    numerical scales, and unpacking the solved vector.
    """

    matrix: np.ndarray  # A (N x P), unscaled
    target: np.ndarray  # x (N,), unscaled
    coefficients: tuple[FosterCoefficientDescriptor, ...]  # P descriptors
    row_scales: np.ndarray  # r (N,)
    col_scales: np.ndarray  # s (P,)
    scaled_matrix: np.ndarray  # Ã = diag(1/r) · A · diag(s)
    scaled_target: np.ndarray  # x̃ = diag(1/r) · x


# ---------------------------------------------------------------------------
# Solve result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FosterSolveResult:
    """Result of solving the Foster linear system."""

    # Coefficients (physical units)
    k0: float | None
    k_inf: float | None
    k_residues: tuple[float, ...]
    f_poles_hz: tuple[float, ...]  # Hz, NOT q_m

    # System classification
    n_targets: int
    n_coefficients: int
    system_class: str  # "square" / "overdetermined" / "underdetermined"

    # Conditioning (of scaled system Ã)
    scaled_condition_number: float
    singular_values: tuple[float, ...]
    rank: int

    # Fit quality
    normalized_residual: float
    max_target_error_ohm: float

    # Normalized coefficient diagnostic — over ALL nonzero u_j,
    # including active-bound.  Bound activity is SEPARATE.
    normalized_coefficient_dynamic_range: float | None

    # Bound activity (separate)
    active_lower_bounds: tuple[int, ...]
    active_upper_bounds: tuple[int, ...]

    # Underdetermined diagnostics
    minimum_bounded_residual: float | None
    selected_fit_residual: float | None
    regularization_lambda: float | None
    regularization_used: bool | None
    regularization_method: str | None

    # Solver convergence
    solver_status: str  # scipy status string

    # Coefficient descriptors
    coefficient_descriptors: tuple[FosterCoefficientDescriptor, ...]

    # Overall verdict
    feasible: bool
    reason: str


# ---------------------------------------------------------------------------
# Build linear system
# ---------------------------------------------------------------------------


def build_foster_linear_system(
    f_targets_hz: np.ndarray,
    x_targets: np.ndarray,
    f_poles_hz: np.ndarray,
    enable_k0: bool,
    enable_kinf: bool,
    coefficient_bounds: CoefficientBounds,
    row_scale_epsilon: float = 1e-3,
    row_scale_abs_min: float = 1.0,
) -> FosterLinearSystem:
    """Build the self-describing, pre-scaled Foster linear system.

    All inputs in Hz.  Derives ω = 2π·f and q_m = (2π·f_p)² internally.

    Parameters
    ----------
    f_targets_hz : ndarray, shape (N,)
        Target frequencies in Hz.
    x_targets : ndarray, shape (N,)
        Target reactance values in Ω.
    f_poles_hz : ndarray, shape (M,)
        Pole frequencies in Hz.
    enable_k0 : bool
        Include endpoint capacitor (k₀) column.
    enable_kinf : bool
        Include endpoint inductor (k∞) column.
    coefficient_bounds : CoefficientBounds
        Physical coefficient bounds.
    row_scale_epsilon : float
        Relative floor for row scaling.
    row_scale_abs_min : float
        Absolute floor for row scaling (Ω).

    Returns
    -------
    FosterLinearSystem
    """
    f_targets = np.asarray(f_targets_hz, dtype=np.float64).ravel()
    x_tgt = np.asarray(x_targets, dtype=np.float64).ravel()
    f_poles = np.asarray(f_poles_hz, dtype=np.float64).ravel()

    n = len(f_targets)
    omega = _TWO_PI * f_targets
    q_m = (_TWO_PI * f_poles) ** 2

    # Build columns + descriptors
    columns: list[np.ndarray] = []
    descriptors: list[FosterCoefficientDescriptor] = []

    if enable_k0 and coefficient_bounds.k0_bounds is not None:
        lb, ub = coefficient_bounds.k0_bounds
        col = -1.0 / omega
        columns.append(col)
        scale = math.sqrt(lb * ub)
        descriptors.append(FosterCoefficientDescriptor(CoefficientKind.K0, None, scale, lb, ub))

    if enable_kinf and coefficient_bounds.kinf_bounds is not None:
        lb, ub = coefficient_bounds.kinf_bounds
        col = omega.copy()
        columns.append(col)
        scale = math.sqrt(lb * ub)
        descriptors.append(FosterCoefficientDescriptor(CoefficientKind.K_INF, None, scale, lb, ub))

    for m_idx in range(len(f_poles)):
        lb, ub = coefficient_bounds.km_bounds[m_idx]
        if lb > ub:
            # Infeasible cell — include anyway so solve can report failure
            pass
        col = omega / (q_m[m_idx] - omega**2)
        columns.append(col)
        scale = math.sqrt(max(lb, 1e-30) * max(ub, 1e-30))
        descriptors.append(
            FosterCoefficientDescriptor(CoefficientKind.K_RESIDUE, m_idx, scale, lb, ub)
        )

    p = len(columns)
    matrix = np.zeros((n, 0), dtype=np.float64) if p == 0 else np.column_stack(columns)

    # Row scaling
    x_char = float(np.max(np.abs(x_tgt))) if n > 0 else 0.0
    if x_char == 0.0:
        row_scales = np.full(n, row_scale_abs_min)
    else:
        row_scales = np.maximum(
            np.abs(x_tgt), np.maximum(row_scale_epsilon * x_char, row_scale_abs_min)
        )

    col_scales = np.array([d.scale for d in descriptors], dtype=np.float64)

    # Scaled system
    if p > 0:
        scaled_matrix = (matrix / row_scales[:, None]) * col_scales[None, :]
    else:
        scaled_matrix = np.zeros((n, 0), dtype=np.float64)
    scaled_target = x_tgt / row_scales

    return FosterLinearSystem(
        matrix=matrix,
        target=x_tgt,
        coefficients=tuple(descriptors),
        row_scales=row_scales,
        col_scales=col_scales,
        scaled_matrix=scaled_matrix,
        scaled_target=scaled_target,
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

_COND_THRESHOLD = 1e12


def solve_foster_system(
    system: FosterLinearSystem,
    f_poles_hz: np.ndarray,
    regularization_lambda: float = 1e-6,
    r_degradation_abs_tol: float = 1e-10,
    r_degradation_rel_tol: float = 0.01,
    condition_threshold: float = _COND_THRESHOLD,
) -> FosterSolveResult:
    """Solve the Foster interpolation system.

    Two-stage strategy for underdetermined systems:
    Stage A — minimum bounded residual.
    Stage B — regularized selection (accepted only if fit not degraded).

    Parameters
    ----------
    system : FosterLinearSystem
        Pre-built linear system.
    f_poles_hz : ndarray
        Pole frequencies in Hz (for result reporting).
    regularization_lambda : float
        Tikhonov regularization weight for underdetermined systems.
    r_degradation_abs_tol : float
        Absolute tolerance for residual degradation guard.
    r_degradation_rel_tol : float
        Relative tolerance for residual degradation guard.
    condition_threshold : float
        Condition number above which the system is flagged ill-conditioned.

    Returns
    -------
    FosterSolveResult
    """
    f_poles_hz = np.asarray(f_poles_hz, dtype=np.float64).ravel()
    a_s = system.scaled_matrix
    x_s = system.scaled_target
    n, p = a_s.shape

    # SVD analysis
    if p == 0:
        # No coefficients — trivial
        return _trivial_result(system, f_poles_hz)

    svd_vals = np.linalg.svd(a_s, compute_uv=False)
    svd_vals_sorted = np.sort(svd_vals)[::-1]
    rank_tol = max(n, p) * np.finfo(float).eps * svd_vals_sorted[0] if svd_vals_sorted[0] > 0 else 0
    rank = int(np.sum(svd_vals_sorted > rank_tol))
    cond = svd_vals_sorted[0] / svd_vals_sorted[-1] if svd_vals_sorted[-1] > 0 else math.inf

    # Bounds in normalized space
    norm_lb = np.array([d.lower_bound / d.scale for d in system.coefficients])
    norm_ub = np.array([d.upper_bound / d.scale for d in system.coefficients])

    # Classify
    if p == n:
        sys_class = "square"
    elif p < n:
        sys_class = "overdetermined"
    else:
        sys_class = "underdetermined"

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    min_bounded_res: float | None = None
    selected_fit_res: float | None = None
    reg_used: bool | None = None
    reg_lambda: float | None = None
    solver_status = ""

    if sys_class == "square" and cond < condition_threshold:
        # Try exact solve
        try:
            u_exact = np.linalg.solve(a_s, x_s)
            k_phys = u_exact * system.col_scales
            bounds_ok = np.all(
                (k_phys >= np.array([d.lower_bound for d in system.coefficients]))
                & (k_phys <= np.array([d.upper_bound for d in system.coefficients]))
            )
            if bounds_ok:
                residual = a_s @ u_exact - x_s
                solver_status = "exact_solve"
                return _build_result(
                    u_exact,
                    residual,
                    system,
                    f_poles_hz,
                    sys_class,
                    cond,
                    svd_vals_sorted,
                    rank,
                    solver_status,
                    min_bounded_res,
                    selected_fit_res,
                    None,
                    None,
                )
        except np.linalg.LinAlgError:
            pass
        # Fall through to bounded LS

    if sys_class != "underdetermined":
        # Square (with bound violations or ill-conditioned) or overdetermined
        result_ls = lsq_linear(a_s, x_s, bounds=(norm_lb, norm_ub), method="trf")
        u_sol = result_ls.x
        residual = a_s @ u_sol - x_s
        solver_status = f"lsq_linear: {result_ls.message}"

        if not _is_lsq_converged(result_ls):
            return _build_result(
                u_sol,
                residual,
                system,
                f_poles_hz,
                sys_class,
                cond,
                svd_vals_sorted,
                rank,
                solver_status,
                None,
                None,
                None,
                None,
                feasible_override=False,
                reason_override=f"Bounded LS did not converge: {result_ls.message}",
            )

        return _build_result(
            u_sol,
            residual,
            system,
            f_poles_hz,
            sys_class,
            cond,
            svd_vals_sorted,
            rank,
            solver_status,
            None,
            None,
            None,
            None,
        )

    # -------------------------------------------------------------------
    # Underdetermined: two-stage solve
    # -------------------------------------------------------------------
    # Stage A: minimum bounded residual
    result_a = lsq_linear(a_s, x_s, bounds=(norm_lb, norm_ub), method="trf")
    u_a = result_a.x
    res_a = np.linalg.norm(a_s @ u_a - x_s)
    min_bounded_res = float(res_a)
    status_a = f"Stage A: {result_a.message}"

    if not _is_lsq_converged(result_a):
        residual = a_s @ u_a - x_s
        return _build_result(
            u_a,
            residual,
            system,
            f_poles_hz,
            sys_class,
            cond,
            svd_vals_sorted,
            rank,
            status_a,
            min_bounded_res,
            float(res_a),
            None,
            False,
            feasible_override=False,
            reason_override=f"Stage A bounded LS did not converge: {result_a.message}",
        )

    # Stage B: regularized selection
    sqrt_lambda = math.sqrt(regularization_lambda)
    a_aug = np.vstack([a_s, sqrt_lambda * np.eye(p)])
    x_aug = np.concatenate([x_s, np.zeros(p)])
    norm_lb_aug = norm_lb  # bounds same
    norm_ub_aug = norm_ub

    result_b = lsq_linear(a_aug, x_aug, bounds=(norm_lb_aug, norm_ub_aug), method="trf")
    u_b = result_b.x
    res_b_fit = float(np.linalg.norm(a_s @ u_b - x_s))  # fit residual only
    status_b = f"Stage B: {result_b.message}"

    b_converged = _is_lsq_converged(result_b)

    # Degradation guard
    accept_reg = b_converged and (
        res_b_fit
        <= min_bounded_res + r_degradation_abs_tol + r_degradation_rel_tol * min_bounded_res
    )

    if accept_reg:
        u_final = u_b
        residual = a_s @ u_b - x_s
        selected_fit_res = res_b_fit
        reg_used = True
        reg_lambda = regularization_lambda
        solver_status = status_b
    else:
        u_final = u_a
        residual = a_s @ u_a - x_s
        selected_fit_res = float(res_a)
        reg_used = False
        reg_lambda = regularization_lambda
        solver_status = status_a + " (Stage B rejected)"

    return _build_result(
        u_final,
        residual,
        system,
        f_poles_hz,
        sys_class,
        cond,
        svd_vals_sorted,
        rank,
        solver_status,
        min_bounded_res,
        selected_fit_res,
        reg_lambda,
        reg_used,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_lsq_converged(result: object) -> bool:
    """Check scipy lsq_linear convergence from its status field."""
    # scipy.optimize.lsq_linear sets .status:
    #  -1 = error
    #   0 = max iterations exceeded
    #   1 = g_norm < gtol (converged)
    #   2 = dx_norm < xtol (converged)
    #   3 = f_norm < ftol (converged)
    status_val = getattr(result, "status", None)
    if status_val is None:
        return True  # defensive fallback
    return int(status_val) in (1, 2, 3)


def _trivial_result(
    system: FosterLinearSystem,
    f_poles_hz: np.ndarray,
) -> FosterSolveResult:
    """Result for a zero-column system."""
    n = len(system.target)
    res_norm = float(np.linalg.norm(system.scaled_target))
    x_norm = float(np.linalg.norm(system.scaled_target))
    norm_res = res_norm if x_norm == 0 else res_norm / x_norm
    max_err = float(np.max(np.abs(system.target))) if n > 0 else 0.0

    return FosterSolveResult(
        k0=None,
        k_inf=None,
        k_residues=(),
        f_poles_hz=tuple(f_poles_hz.tolist()),
        n_targets=n,
        n_coefficients=0,
        system_class="trivial",
        scaled_condition_number=0.0,
        singular_values=(),
        rank=0,
        normalized_residual=norm_res,
        max_target_error_ohm=max_err,
        normalized_coefficient_dynamic_range=None,
        active_lower_bounds=(),
        active_upper_bounds=(),
        minimum_bounded_residual=None,
        selected_fit_residual=None,
        regularization_lambda=None,
        regularization_used=None,
        regularization_method=None,
        solver_status="trivial_no_columns",
        coefficient_descriptors=system.coefficients,
        feasible=max_err < 1e-10,
        reason="trivial" if max_err < 1e-10 else "nonzero target with no coefficients",
    )


def _build_result(
    u: np.ndarray,
    residual: np.ndarray,
    system: FosterLinearSystem,
    f_poles_hz: np.ndarray,
    sys_class: str,
    cond: float,
    sv: np.ndarray,
    rank: int,
    solver_status: str,
    min_bounded_res: float | None,
    selected_fit_res: float | None,
    reg_lambda: float | None,
    reg_used: bool | None,
    *,
    feasible_override: bool | None = None,
    reason_override: str | None = None,
) -> FosterSolveResult:
    """Construct FosterSolveResult from solved normalized vector u."""
    # Unpack coefficients
    k_phys = u * system.col_scales
    k0_val: float | None = None
    kinf_val: float | None = None
    k_residues: list[float] = []

    for desc, kp in zip(system.coefficients, k_phys, strict=True):
        if desc.kind == CoefficientKind.K0:
            k0_val = float(kp)
        elif desc.kind == CoefficientKind.K_INF:
            kinf_val = float(kp)
        elif desc.kind == CoefficientKind.K_RESIDUE:
            k_residues.append(float(kp))

    # Physical-space residual
    x_fit = system.matrix @ k_phys
    phys_err = np.abs(x_fit - system.target)
    max_err = float(np.max(phys_err)) if len(phys_err) > 0 else 0.0

    # Normalized residual (zero-target safety)
    x_norm = float(np.linalg.norm(system.scaled_target))
    res_norm = float(np.linalg.norm(residual))
    norm_res = res_norm / x_norm if x_norm > 0 else res_norm  # no div by zero

    # Normalized coefficient dynamic range — ALL nonzero u_j
    u_abs = np.abs(u)
    nonzero_mask = u_abs > 1e-30
    if np.any(nonzero_mask):
        u_nonzero = u_abs[nonzero_mask]
        dyn_range: float | None = float(np.max(u_nonzero) / np.min(u_nonzero))
    else:
        dyn_range = None

    # Bound activity
    active_lb: list[int] = []
    active_ub: list[int] = []
    bound_tol = 1e-8
    for j, desc in enumerate(system.coefficients):
        norm_lb = desc.lower_bound / desc.scale
        norm_ub = desc.upper_bound / desc.scale
        if abs(u[j] - norm_lb) < bound_tol * max(1.0, abs(norm_lb)):
            active_lb.append(j)
        if abs(u[j] - norm_ub) < bound_tol * max(1.0, abs(norm_ub)):
            active_ub.append(j)

    # Feasibility
    bounds_ok = all(
        desc.lower_bound <= k_phys[j] <= desc.upper_bound
        for j, desc in enumerate(system.coefficients)
    )
    residual_ok = max_err < 1.0  # generous for seed candidacy
    feasible = bounds_ok and residual_ok
    reason = "ok"
    if not bounds_ok:
        reason = "coefficient(s) outside physical bounds"
    elif not residual_ok:
        reason = f"max target error {max_err:.4g} Ω"

    if feasible_override is not None:
        feasible = feasible_override
    if reason_override is not None:
        reason = reason_override

    return FosterSolveResult(
        k0=k0_val,
        k_inf=kinf_val,
        k_residues=tuple(k_residues),
        f_poles_hz=tuple(f_poles_hz.tolist()),
        n_targets=len(system.target),
        n_coefficients=len(system.coefficients),
        system_class=sys_class,
        scaled_condition_number=float(cond),
        singular_values=tuple(sv.tolist()),
        rank=rank,
        normalized_residual=norm_res,
        max_target_error_ohm=max_err,
        normalized_coefficient_dynamic_range=dyn_range,
        active_lower_bounds=tuple(active_lb),
        active_upper_bounds=tuple(active_ub),
        minimum_bounded_residual=min_bounded_res,
        selected_fit_residual=selected_fit_res,
        regularization_lambda=reg_lambda,
        regularization_used=reg_used,
        regularization_method="augmented_system" if reg_used else None,
        solver_status=solver_status,
        coefficient_descriptors=system.coefficients,
        feasible=feasible,
        reason=reason,
    )
