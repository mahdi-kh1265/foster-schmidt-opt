"""Local polish runner for Prompt 05.

Polishes basin representatives using ``trust-constr`` (preferred) or the
configured fallback.  SLSQP is rejected.  Local budget is per-basin.

Pre-polish retention: if the polished result is Deb-worse than the pre-polish
result, the pre-polish is retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foster_eom.domain.objectives import LocalMethod, OptimizationSpec
from foster_eom.optimize.dedup import Basin, deb_better, deb_key
from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    EvaluationContext,
    EvaluationResult,
    evaluate,
)

# ---------------------------------------------------------------------------
# Polish result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolishResult:
    """Result of polishing one basin representative."""

    domain_id: str
    basin_index: int
    pre_polish: EvaluationResult
    post_polish: EvaluationResult
    retained: EvaluationResult         # the better of pre/post
    method_used: str
    success: bool
    n_iterations: int
    n_evaluations: int
    termination: str
    reason: str | None                 # why we fell back or failed


# ---------------------------------------------------------------------------
# Polish one basin
# ---------------------------------------------------------------------------


def _resolve_method(opt_spec: OptimizationSpec) -> str:
    """Determine the local solver method to use."""
    primary = opt_spec.local_method
    fallback = opt_spec.local_fallback_method

    if primary == LocalMethod.SLSQP:
        raise RuntimeError("SLSQP should have been rejected in preflight")

    if primary == LocalMethod.IPOPT:
        try:
            import cyipopt  # type: ignore[import]
            return "ipopt"
        except ImportError:
            pass  # fall through to fallback
        method = fallback
    else:
        method = primary

    if method == LocalMethod.TRUST_CONSTR:
        return "trust-constr"
    if method == LocalMethod.IPOPT:
        return "trust-constr"  # last resort
    return "trust-constr"


def polish_basin(
    basin: Basin,
    basin_index: int,
    context: EvaluationContext,
    cache: DomainEvaluatorCache,
    opt_spec: OptimizationSpec,
) -> PolishResult:
    """Polish the representative of one basin.

    Uses forward finite differences in normalized [0,1]^n space.
    If the polished result is Deb-worse than the pre-polish result, the
    pre-polish result is retained.
    """
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    pre = basin.representative
    domain_id = context.domain.domain_id
    n_dim = context.domain.dimension

    if n_dim == 0:
        return PolishResult(
            domain_id=domain_id,
            basin_index=basin_index,
            pre_polish=pre,
            post_polish=pre,
            retained=pre,
            method_used="none",
            success=True,
            n_iterations=0,
            n_evaluations=0,
            termination="zero_dimensional",
            reason=None,
        )

    method_str = _resolve_method(opt_spec)

    x0 = np.array(pre.x, dtype=np.float64)
    bounds = Bounds(lb=0.0, ub=1.0)
    fd_step = opt_spec.finite_difference_step
    max_iter = opt_spec.local_max_iterations

    n_evals_before = cache.n_unique_evaluations

    def _obj(x: np.ndarray) -> float:
        return evaluate(x, context, cache).objective_value

    def _g_vec(x: np.ndarray) -> np.ndarray:
        r = evaluate(x, context, cache)
        if not r.hard_margins:
            return np.array([1.0])
        return np.array(r.hard_margins, dtype=np.float64)

    nlc = NonlinearConstraint(_g_vec, lb=0.0, ub=np.inf)

    try:
        scipy_result = minimize(
            fun=_obj,
            x0=x0,
            method=method_str,
            bounds=bounds,
            constraints=nlc,
            jac="2-point",
            options={
                "maxiter": max_iter,
                "finite_diff_rel_step": fd_step,
                "verbose": 0,
            },
        )
        post_x = np.clip(scipy_result.x, 0.0, 1.0)
        post_result = evaluate(post_x, context, cache)
        success = bool(scipy_result.success)
        n_iter = getattr(scipy_result, "nit", 0)
        term_msg = scipy_result.message
        reason = None
    except Exception as exc:
        # Polish failed — retain pre-polish
        post_result = pre
        success = False
        n_iter = 0
        term_msg = f"exception: {type(exc).__name__}"
        reason = str(exc)

    n_evals = cache.n_unique_evaluations - n_evals_before

    # Pre-polish retention: if post is Deb-worse, discard
    if deb_better(pre, post_result):
        retained = pre
    else:
        retained = post_result

    return PolishResult(
        domain_id=domain_id,
        basin_index=basin_index,
        pre_polish=pre,
        post_polish=post_result,
        retained=retained,
        method_used=method_str,
        success=success,
        n_iterations=n_iter,
        n_evaluations=n_evals,
        termination=term_msg,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Polish top-K basins
# ---------------------------------------------------------------------------


def polish_top_k(
    basins: list[Basin],
    context: EvaluationContext,
    cache: DomainEvaluatorCache,
    opt_spec: OptimizationSpec,
) -> list[PolishResult]:
    """Polish the top-k basin representatives (by Deb key of representative)."""
    k = opt_spec.polish_top_k
    sorted_basins = sorted(basins, key=lambda b: deb_key(b.representative))
    selected = sorted_basins[:k]

    results: list[PolishResult] = []
    for i, basin in enumerate(selected):
        pr = polish_basin(basin, i, context, cache, opt_spec)
        results.append(pr)

    return results
