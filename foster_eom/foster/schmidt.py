"""Schmidt target-reactance solver (Prompt 04A).

Implements the standard and admittance-dual L-network target equations
with dimensionless gap-free tolerance classification, explicit degenerate
handling, and typed reactance target representation.

All public APIs accept frequencies in Hz.
"""
# NOTE: Mathematical comments in this file use ASCII approximations
# of Greek letters: rho (r), epsilon (eps), omega (w).

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from foster_eom.domain.topology import LOrientation

# ---------------------------------------------------------------------------
# Enums and tolerance structures
# ---------------------------------------------------------------------------


class ReactanceTargetState(enum.StrEnum):
    """Whether a reactance target is finite or an open circuit."""

    FINITE = "finite"
    OPEN_CIRCUIT = "open_circuit"


class TargetFeasibility(enum.StrEnum):
    """Per-frequency feasibility classification."""

    ORDINARY = "ordinary"
    DEGENERATE = "degenerate"
    INFEASIBLE = "infeasible"


class BranchRealization(enum.StrEnum):
    """Physical realization of a Foster branch.

    FINITE_FOSTER  — Foster LC network with >= 1 component.
    ZERO_IMPEDANCE — Wire / short (zero-impedance path).  Valid only as
                     series branch; illegal as shunt (shorts RF port).
    OPEN_OMITTED   — Branch physically absent.  Valid only as shunt;
                     illegal as series (disconnects load).
    """

    FINITE_FOSTER = "finite_foster"
    ZERO_IMPEDANCE = "zero_impedance"
    OPEN_OMITTED = "open_omitted"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactanceTarget:
    """A single-frequency reactance target for one Foster branch.

    Invariants
    ----------
    * FINITE   → value_ohm is a finite float.
    * OPEN_CIRCUIT → value_ohm is None.
    """

    f_hz: float
    value_ohm: float | None
    state: ReactanceTargetState

    def __post_init__(self) -> None:
        if self.state == ReactanceTargetState.FINITE:
            if self.value_ohm is None or not math.isfinite(self.value_ohm):
                raise ValueError(
                    f"FINITE target requires a finite float value_ohm, got {self.value_ohm!r}"
                )
        elif self.state == ReactanceTargetState.OPEN_CIRCUIT and self.value_ohm is not None:
            raise ValueError(
                f"OPEN_CIRCUIT target requires value_ohm = None, got {self.value_ohm!r}"
            )


@dataclass(frozen=True)
class SchmidtTolerances:
    """Dimensionless floating-point classification tolerances.

    epsilon_rho : standard orientation -- |rho - 1| <= eps classifies degenerate.
    epsilon_g   : dual orientation    -- |g - 1| <= eps classifies degenerate.
    bp_rel_tol  : |B_p| * R_match < tol -- OPEN parallel element.
    """

    epsilon_rho: float = 1e-10
    epsilon_g: float = 1e-10
    bp_rel_tol: float = 1e-12


@dataclass(frozen=True)
class FosterBranchTolerances:
    """Tolerances for branch realization classification.

    Zero-reactance tolerance is relative to R_match (a physically
    independent scale), not to the branch's own max |X|.

    A branch is classified ZERO_IMPEDANCE iff every FINITE target
    satisfies |X_i| ≤ x_zero_abs + x_zero_rel · R_match.
    """

    x_zero_abs: float = 0.01  # Ω — absolute threshold
    x_zero_rel: float = 1e-6  # relative to R_match


_DEFAULT_TOLERANCES = SchmidtTolerances()
_DEFAULT_BRANCH_TOL = FosterBranchTolerances()


@dataclass(frozen=True)
class SchmidtTargetPoint:
    """Per-frequency Schmidt target for one orientation."""

    f_hz: float
    z_load: complex
    rho_or_g: float  # rho = R_L/R_match (standard) or g = G*R_match (dual)
    x_shunt_plus: ReactanceTarget
    x_shunt_minus: ReactanceTarget
    x_series_for_plus: ReactanceTarget
    x_series_for_minus: ReactanceTarget
    feasibility: TargetFeasibility
    failure_reason: str | None


@dataclass(frozen=True)
class SchmidtResult:
    """Full Schmidt target result for one orientation."""

    r_match_ohm: float
    orientation: LOrientation
    points: tuple[SchmidtTargetPoint, ...]
    all_valid: bool
    tolerances: SchmidtTolerances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(f_hz: float, value: float) -> ReactanceTarget:
    return ReactanceTarget(f_hz, value, ReactanceTargetState.FINITE)


def _open(f_hz: float) -> ReactanceTarget:
    return ReactanceTarget(f_hz, None, ReactanceTargetState.OPEN_CIRCUIT)


# ---------------------------------------------------------------------------
# Standard orientation
# ---------------------------------------------------------------------------


def schmidt_standard_targets(
    r_match_ohm: float,
    z_load: np.ndarray,
    f_hz: np.ndarray,
    tolerances: SchmidtTolerances | None = None,
) -> SchmidtResult:
    """Standard Schmidt orientation (shunt X₂, series X₁).

    Topology::

        source → [in] → shunt jX₂ to [gnd]
                      → series jX₁ → [eom_pos] → Z_EOM → [gnd]

    Feasibility:  0 < R_L ≤ R_match.
    Degenerate:   R_L ~ R_match -> open shunt, X1 = -X_L.

    Parameters
    ----------
    r_match_ohm : float
        Desired real input match resistance.
    z_load : np.ndarray
        Complex load impedances, shape (N,).
    f_hz : np.ndarray
        Target frequencies in Hz, shape (N,).
    tolerances : SchmidtTolerances | None
        Classification tolerances.  Defaults used if None.

    Returns
    -------
    SchmidtResult
    """
    if r_match_ohm <= 0.0:
        raise ValueError(f"r_match_ohm must be positive, got {r_match_ohm}")
    tol = tolerances or _DEFAULT_TOLERANCES
    z_load = np.asarray(z_load, dtype=np.complex128).ravel()
    f_hz = np.asarray(f_hz, dtype=np.float64).ravel()
    if z_load.shape != f_hz.shape:
        raise ValueError("z_load and f_hz must have the same length")

    points: list[SchmidtTargetPoint] = []
    for i in range(len(f_hz)):
        fi = float(f_hz[i])
        zl = complex(z_load[i])
        rl = zl.real
        xl = zl.imag
        rho = rl / r_match_ohm

        if rl <= 0.0:
            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=rho,
                x_shunt_plus=_open(fi),
                x_shunt_minus=_open(fi),
                x_series_for_plus=_open(fi),
                x_series_for_minus=_open(fi),
                feasibility=TargetFeasibility.INFEASIBLE,
                failure_reason=f"R_L = {rl:.6g} ≤ 0",
            )
        elif abs(rho - 1.0) <= tol.epsilon_rho:
            # Degenerate: shunt open, series cancels reactance
            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=rho,
                x_shunt_plus=_open(fi),
                x_shunt_minus=_open(fi),
                x_series_for_plus=_finite(fi, -xl),
                x_series_for_minus=_finite(fi, -xl),
                feasibility=TargetFeasibility.DEGENERATE,
                failure_reason=None,
            )
        elif rho > 1.0 + tol.epsilon_rho:
            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=rho,
                x_shunt_plus=_open(fi),
                x_shunt_minus=_open(fi),
                x_series_for_plus=_open(fi),
                x_series_for_minus=_open(fi),
                feasibility=TargetFeasibility.INFEASIBLE,
                failure_reason=f"rho = {rho:.6g} > 1 + eps (R_L > R_match)",
            )
        else:
            # Ordinary: 0 < rho < 1 - eps  (within tolerance)
            radicand = rho * (1.0 - rho)
            # This should be positive since rho < 1 - eps.  A negative value
            # here would indicate a numerical/logic error.
            if radicand < 0.0:
                raise RuntimeError(
                    f"Numerical error: ordinary-classified point has negative "
                    f"radicand rho(1-rho) = {radicand:.6e} at rho = {rho:.15e}.  "
                    f"This should not happen after gap-free classification."
                )
            # X2 = +/- R_match * sqrt(rho / (1 - rho))
            x2_mag = r_match_ohm * math.sqrt(rho / (1.0 - rho))
            x2_plus = x2_mag
            x2_minus = -x2_mag

            # X1 = -R_match^2 * X2 / (R_match^2 + X2^2) - X_L
            rm2 = r_match_ohm * r_match_ohm
            x1_for_plus = -rm2 * x2_plus / (rm2 + x2_plus**2) - xl
            x1_for_minus = -rm2 * x2_minus / (rm2 + x2_minus**2) - xl

            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=rho,
                x_shunt_plus=_finite(fi, x2_plus),
                x_shunt_minus=_finite(fi, x2_minus),
                x_series_for_plus=_finite(fi, x1_for_plus),
                x_series_for_minus=_finite(fi, x1_for_minus),
                feasibility=TargetFeasibility.ORDINARY,
                failure_reason=None,
            )
        points.append(pt)

    all_valid = all(p.feasibility != TargetFeasibility.INFEASIBLE for p in points)
    return SchmidtResult(
        r_match_ohm=r_match_ohm,
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        points=tuple(points),
        all_valid=all_valid,
        tolerances=tol,
    )


# ---------------------------------------------------------------------------
# Dual (admittance) orientation
# ---------------------------------------------------------------------------


def schmidt_dual_targets(
    r_match_ohm: float,
    z_load: np.ndarray,
    f_hz: np.ndarray,
    tolerances: SchmidtTolerances | None = None,
) -> SchmidtResult:
    """Admittance-dual orientation (series X_s, shunt X_p).

    Topology::

        source → [in] → series jX_s → [mid] → Z_EOM
                                        |
                                        +→ shunt jX_p → [gnd]

    Feasibility: 0 < G ≤ 1/R_match.
    Degenerate:  G ≈ 1/R_match → zero series, parallel cancels susceptance.

    Parameters
    ----------
    r_match_ohm : float
        Desired real input match resistance.
    z_load : np.ndarray
        Complex load impedances, shape (N,).
    f_hz : np.ndarray
        Target frequencies in Hz, shape (N,).
    tolerances : SchmidtTolerances | None
        Classification tolerances.

    Returns
    -------
    SchmidtResult
    """
    if r_match_ohm <= 0.0:
        raise ValueError(f"r_match_ohm must be positive, got {r_match_ohm}")
    tol = tolerances or _DEFAULT_TOLERANCES
    z_load = np.asarray(z_load, dtype=np.complex128).ravel()
    f_hz = np.asarray(f_hz, dtype=np.float64).ravel()
    if z_load.shape != f_hz.shape:
        raise ValueError("z_load and f_hz must have the same length")

    points: list[SchmidtTargetPoint] = []
    for i in range(len(f_hz)):
        fi = float(f_hz[i])
        zl = complex(z_load[i])
        # Admittance of load
        yl = 1.0 / zl
        g_load = yl.real
        b_load = yl.imag
        g = g_load * r_match_ohm  # dimensionless

        if g_load <= 0.0:
            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=g,
                x_shunt_plus=_open(fi),
                x_shunt_minus=_open(fi),
                x_series_for_plus=_open(fi),
                x_series_for_minus=_open(fi),
                feasibility=TargetFeasibility.INFEASIBLE,
                failure_reason=f"G = {g_load:.6g} ≤ 0",
            )
        elif abs(g - 1.0) <= tol.epsilon_g:
            # Degenerate: X_s = 0, parallel cancels susceptance
            # X_p = 1/B to cancel B_p = -B, or OPEN if B ~= 0
            xp = 1.0 / b_load if abs(b_load) * r_match_ohm > tol.bp_rel_tol else None

            xs_target = _finite(fi, 0.0)
            xp_target = _open(fi) if xp is None else _finite(fi, xp)

            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=g,
                x_shunt_plus=xp_target,
                x_shunt_minus=xp_target,
                x_series_for_plus=xs_target,
                x_series_for_minus=xs_target,
                feasibility=TargetFeasibility.DEGENERATE,
                failure_reason=None,
            )
        elif g > 1.0 + tol.epsilon_g:
            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=g,
                x_shunt_plus=_open(fi),
                x_shunt_minus=_open(fi),
                x_series_for_plus=_open(fi),
                x_series_for_minus=_open(fi),
                feasibility=TargetFeasibility.INFEASIBLE,
                failure_reason=f"g = {g:.6g} > 1 + eps (G > 1/R_match)",
            )
        else:
            # Ordinary: 0 < g < 1 - eps
            radicand = g * (1.0 - g)
            if radicand < 0.0:
                raise RuntimeError(
                    f"Numerical error: ordinary-classified dual point has "
                    f"negative radicand g(1-g) = {radicand:.6e} at g = {g:.15e}."
                )
            # B_t = +/- sqrt(g(1-g)) / R_match
            bt_mag = math.sqrt(radicand) / r_match_ohm
            bt_plus = bt_mag
            bt_minus = -bt_mag

            def _dual_pair(
                bt: float,
                _b_load: float = b_load,
                _g_load: float = g_load,
                _fi: float = fi,
            ) -> tuple[ReactanceTarget, ReactanceTarget]:
                bp = bt - _b_load
                xs = r_match_ohm * bt / _g_load
                if abs(bp) * r_match_ohm < tol.bp_rel_tol:
                    xp_t = _open(_fi)
                else:
                    xp_t = _finite(_fi, -1.0 / bp)
                xs_t = _finite(_fi, xs)
                return xp_t, xs_t

            xp_plus, xs_plus = _dual_pair(bt_plus)
            xp_minus, xs_minus = _dual_pair(bt_minus)

            pt = SchmidtTargetPoint(
                f_hz=fi,
                z_load=zl,
                rho_or_g=g,
                x_shunt_plus=xp_plus,
                x_shunt_minus=xp_minus,
                x_series_for_plus=xs_plus,
                x_series_for_minus=xs_minus,
                feasibility=TargetFeasibility.ORDINARY,
                failure_reason=None,
            )
        points.append(pt)

    all_valid = all(p.feasibility != TargetFeasibility.INFEASIBLE for p in points)
    return SchmidtResult(
        r_match_ohm=r_match_ohm,
        orientation=LOrientation.ALTERNATE_L_ORIENTATION,
        points=tuple(points),
        all_valid=all_valid,
        tolerances=tol,
    )


# ---------------------------------------------------------------------------
# Algebraic validation
# ---------------------------------------------------------------------------


def validate_schmidt_targets_algebraic(
    result: SchmidtResult,
    signs: tuple[int, ...],
    tol: float = 1e-8,
) -> tuple[bool, tuple[complex, ...]]:
    """Plug X₁,X₂ into L-network formula, verify Z_in ≈ R_match.

    After tolerance-snapping to degenerate, uses compatible tolerance.

    Parameters
    ----------
    result : SchmidtResult
        Schmidt target result.
    signs : tuple[int, ...]
        +1 or -1 per frequency, selecting the sign branch.
    tol : float
        Relative error tolerance: |Z_in - R_match| / R_match < tol.

    Returns
    -------
    (all_ok, z_in_values) : tuple[bool, tuple[complex, ...]]
    """
    rm = result.r_match_ohm
    z_in_list: list[complex] = []
    all_ok = True

    for _idx, (pt, sign) in enumerate(zip(result.points, signs, strict=True)):
        if pt.feasibility == TargetFeasibility.INFEASIBLE:
            z_in_list.append(complex("nan"))
            all_ok = False
            continue

        # Select the sign branch targets
        if sign >= 0:
            x_shunt_t = pt.x_shunt_plus
            x_series_t = pt.x_series_for_plus
        else:
            x_shunt_t = pt.x_shunt_minus
            x_series_t = pt.x_series_for_minus

        zl = pt.z_load

        if result.orientation == LOrientation.SCHMIDT_SHUNT_THEN_SERIES:
            # Standard: Z_in = jX₂ ‖ (jX₁ + Z_L)
            x1 = x_series_t.value_ohm if x_series_t.state == ReactanceTargetState.FINITE else None
            x2 = x_shunt_t.value_ohm if x_shunt_t.state == ReactanceTargetState.FINITE else None

            if x1 is None:
                # Series branch open — shouldn't happen in valid target
                z_in = complex("nan")
            elif x2 is None:
                # Shunt open: Z_in = jX₁ + Z_L
                z_in = 1j * x1 + zl
            else:
                # Z_in = jX₂ · (jX₁ + Z_L) / (jX₂ + jX₁ + Z_L)
                jx1 = 1j * x1
                jx2 = 1j * x2
                z_in = jx2 * (jx1 + zl) / (jx2 + jx1 + zl)
        else:
            # Dual: Z_in = jX_s + (Z_L ‖ jX_p)
            xs = x_series_t.value_ohm if x_series_t.state == ReactanceTargetState.FINITE else None
            xp = x_shunt_t.value_ohm if x_shunt_t.state == ReactanceTargetState.FINITE else None

            xs_val = 0.0 if xs is None else xs  # Degenerate: zero series

            if xp is None:
                # Parallel open: Z_in = jX_s + Z_L
                z_in = 1j * xs_val + zl
            else:
                jxp = 1j * xp
                z_parallel = jxp * zl / (jxp + zl)
                z_in = 1j * xs_val + z_parallel

        z_in_list.append(z_in)
        err = abs(z_in - rm) / rm
        if err > tol:
            all_ok = False

    return all_ok, tuple(z_in_list)


# ---------------------------------------------------------------------------
# Branch realization classification
# ---------------------------------------------------------------------------


def classify_branch_realization(
    targets: tuple[ReactanceTarget, ...],
    r_match_ohm: float,
    is_series: bool,
    branch_tol: FosterBranchTolerances | None = None,
) -> BranchRealization:
    """Classify a branch's target list into a realization state.

    Parameters
    ----------
    targets : tuple[ReactanceTarget, ...]
        The reactance targets for one branch.
    r_match_ohm : float
        Match resistance — used as the physically independent scale
        for zero-reactance classification.
    is_series : bool
        True if the branch is series, False if shunt.
    branch_tol : FosterBranchTolerances | None
        Tolerances for zero classification.

    Returns
    -------
    BranchRealization

    Raises
    ------
    ValueError
        If the branch has mixed OPEN + FINITE targets (structurally
        infeasible in Prompt 04).
    """
    bt = branch_tol or _DEFAULT_BRANCH_TOL

    has_finite = any(t.state == ReactanceTargetState.FINITE for t in targets)
    has_open = any(t.state == ReactanceTargetState.OPEN_CIRCUIT for t in targets)

    if has_open and has_finite:
        raise ValueError(
            "Mixed OPEN + FINITE targets are structurally infeasible "
            "in Prompt 04.  Exact Foster realization requires a pole "
            "at the OPEN target frequency, which is prohibited."
        )

    if not has_finite:
        # All OPEN
        return BranchRealization.OPEN_OMITTED

    # All FINITE — check if all effectively zero
    zero_threshold = bt.x_zero_abs + bt.x_zero_rel * r_match_ohm
    all_zero = all(
        abs(t.value_ohm) <= zero_threshold  # type: ignore[arg-type]
        for t in targets
        if t.state == ReactanceTargetState.FINITE
    )
    if all_zero:
        if is_series:
            return BranchRealization.ZERO_IMPEDANCE
        else:
            # For shunt, only exactly 0.0 becomes ZERO_IMPEDANCE (which is illegal).
            # Small finite values remain FINITE_FOSTER to avoid false infeasibility.
            exact_zero = all(
                t.value_ohm == 0.0 for t in targets if t.state == ReactanceTargetState.FINITE
            )
            if exact_zero:
                return BranchRealization.ZERO_IMPEDANCE
            return BranchRealization.FINITE_FOSTER

    return BranchRealization.FINITE_FOSTER


def validate_branch_realization_legality(
    realization: BranchRealization,
    is_series: bool,
) -> tuple[bool, str | None]:
    """Check whether a branch realization is legal for its position.

    Returns (legal, reason).
    """
    if realization == BranchRealization.ZERO_IMPEDANCE and not is_series:
        return False, "ZERO_IMPEDANCE shunt branch would short input to ground"
    if realization == BranchRealization.OPEN_OMITTED and is_series:
        return False, "OPEN_OMITTED series branch would disconnect load"
    return True, None
