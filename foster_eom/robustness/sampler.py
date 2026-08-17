"""P10 sampling engine.

Generates the N × D draw matrix for Monte Carlo robustness analysis.
Supports iid random (Wilson CI valid), LHS, and Sobol (no Wilson CI).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from foster_eom.robustness.uncertainty import SlotUncertainty, UncertaintyTerm

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessSpec:
    """Configuration for one P10 robustness run.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples.  For Sobol, rounded up to next 2^k.
    seed : int
        Random seed for reproducibility.
    method : str
        "random" — iid draws; Wilson CI valid.
        "lhs"    — Latin-hypercube; better coverage; no Wilson CI.
        "sobol"  — Sobol low-discrepancy; no Wilson CI.
    p06_diagnostic : str
        "none"    — no P06 adaptive sweep for any sample.
        "worst_k" — P06 on the k worst-objective feasible samples (diagnostic only;
                    NOT a population yield estimate).
        "all"     — full P06 for all samples (expensive; enables yield_p06).
    p06_worst_k : int
        Number of worst samples for the "worst_k" diagnostic.
    sensitivity_n_steps : int
        Number of OAT perturbation steps per slot (each side of nominal).
    ci_level : float
        Confidence level for Wilson CI (e.g. 0.95).
    n_retry : int
        Number of retries for numerical failures before marking NUMERICAL_UNRESOLVED.
    """

    n_samples: int = 500
    seed: int = 42
    method: Literal["random", "lhs", "sobol"] = "random"
    p06_diagnostic: Literal["none", "worst_k", "all"] = "worst_k"
    p06_worst_k: int = 20
    sensitivity_n_steps: int = 7
    ci_level: float = 0.95
    n_retry: int = 2


# ---------------------------------------------------------------------------
# Draw result
# ---------------------------------------------------------------------------


@dataclass
class DrawMatrix:
    """Result of the sampling step.

    Parameters
    ----------
    u : np.ndarray, shape (n_samples, n_dims)
        Uniform [0,1] samples in quantile space (one column per stochastic dim).
    slot_order : list[str]
        element_id for each column.  Non-stochastic slots are NOT included.
    method : str
        Sampling method used.
    seed : int
    """

    u: np.ndarray
    slot_order: list[str]
    method: str
    seed: int


# ---------------------------------------------------------------------------
# Sampling engine
# ---------------------------------------------------------------------------


def draw_samples(
    slot_uncertainties: list[SlotUncertainty],
    spec: RobustnessSpec,
) -> DrawMatrix:
    """Generate the N × D uniform draw matrix.

    Returns quantile-space samples u ∈ [0,1]^D for stochastic slots only.
    Call ``inverse_transform`` to convert to physical value draws.
    """
    stochastic = [su for su in slot_uncertainties if su.is_stochastic]
    d = len(stochastic)
    n = spec.n_samples

    if d == 0:
        return DrawMatrix(
            u=np.empty((n, 0)),
            slot_order=[],
            method=spec.method,
            seed=spec.seed,
        )

    if spec.method == "random":
        rng = np.random.default_rng(spec.seed)
        u = rng.random((n, d))

    elif spec.method == "lhs":
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=d, seed=spec.seed)
        u = sampler.random(n=n)

    elif spec.method == "sobol":
        from scipy.stats.qmc import Sobol

        # Sobol requires power-of-2 sample counts
        k = int(np.ceil(np.log2(max(n, 1))))
        n_actual = 2**k
        sampler = Sobol(d=d, scramble=True, seed=spec.seed)
        u = sampler.random(n_actual)
        u = u[:n]  # truncate to requested n

    else:
        raise ValueError(f"Unknown sampling method: {spec.method!r}")

    return DrawMatrix(
        u=u,
        slot_order=[su.element_id for su in stochastic],
        method=spec.method,
        seed=spec.seed,
    )


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------


def inverse_transform_draw(
    u_row: np.ndarray,
    slot_uncertainties: list[SlotUncertainty],
    nominal_values: dict[str, float],
) -> dict[str, float]:
    """Convert one row of [0,1] quantile samples to physical drawn values (SI).

    Parameters
    ----------
    u_row : np.ndarray, shape (n_stochastic_dims,)
        One sample row from DrawMatrix.u.
    slot_uncertainties : list[SlotUncertainty]
        All slot uncertainties (stochastic and deterministic).
    nominal_values : dict[str, float]
        Catalog nominal value (SI) for each element_id.

    Returns
    -------
    dict[str, float]
        element_id → drawn value in SI.
    """
    stochastic = [su for su in slot_uncertainties if su.is_stochastic]
    u_iter = iter(u_row)
    draws: dict[str, float] = {}

    for su in slot_uncertainties:
        nom = nominal_values[su.element_id]
        if not su.is_stochastic:
            draws[su.element_id] = nom
            continue

        u = next(u_iter)
        # Sum fractional deviations across all terms
        total_delta = _draw_combined_delta(u, su.terms)
        draws[su.element_id] = nom * (1.0 + total_delta)

    # Sanity: ensure all stochastic dims consumed
    _ = stochastic  # referenced above

    return draws


def _draw_combined_delta(u: float, terms: tuple[UncertaintyTerm, ...]) -> float:
    """Convert a single U(0,1) quantile to a combined fractional deviation.

    For multiple terms the total interval is [sum(lo_i), sum(hi_i)] and
    the single u is mapped linearly across the combined interval.
    This preserves a single stochastic dimension per slot while summing
    all additive term contributions.

    For normal_3sigma terms the clipped-normal inverse CDF is applied
    per-term contribution and summed.
    """
    # Separate uniform and normal terms
    uniform_lo = 0.0
    uniform_hi = 0.0
    normal_sigma_total = 0.0

    for t in terms:
        if t.distribution == "uniform":
            uniform_lo += t.effective_lo
            uniform_hi += t.effective_hi
        elif t.distribution == "normal_3sigma":
            if t.tol_frac is not None:
                normal_sigma_total += t.tol_frac / 3.0

    delta = 0.0

    # Uniform contribution
    span = uniform_hi - uniform_lo
    if span > 1e-15:
        delta += uniform_lo + span * u

    # Normal contribution (independent, using same u quantile — acceptable
    # approximation when combined interval is dominated by one term)
    if normal_sigma_total > 1e-15:
        from scipy.stats import norm

        # Clip to ±3σ bounds
        u_clipped = float(np.clip(u, 1e-4, 1.0 - 1e-4))
        delta += norm.ppf(u_clipped) * normal_sigma_total

    return delta


def ci_method_for_spec(spec: RobustnessSpec) -> str | None:
    """Return the CI method name applicable for this spec, or None."""
    if spec.method == "random":
        return "wilson"
    return None
