"""Local polish runner for Prompt 05.

Polishes basin representatives using ``trust-constr`` (preferred) or the
configured fallback.  SLSQP is rejected.  Local budget is per-basin.

Pre-polish retention: if the polished result is Deb-worse than the pre-polish
result, the pre-polish is retained.

P12.5-E adds an explicit derivative mode.  ``REFERENCE_FD`` is the frozen P05
behaviour (SciPy differentiates the objective and the nonlinear constraints
numerically).  ``ANALYTICAL`` supplies the validated P12.5-D Jacobians instead.
The objective and constraint *values* are the same frozen ``evaluate()``
callbacks in both modes; only ``jac`` differs.  Any unsupported, nonsmooth, or
unresolved derivative state falls the affected candidate back to
``REFERENCE_FD`` — a partial Jacobian is never handed to SciPy.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from foster_eom.domain.objectives import DerivativeMode, LocalMethod, OptimizationSpec
from foster_eom.optimize.dedup import Basin, deb_better, deb_key
from foster_eom.optimize.derivative_provider import (
    AnalyticalDerivativeProvider,
    DerivativeUnavailable,
)
from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    EvaluationContext,
    EvaluationResult,
    evaluate,
)
from foster_eom.optimize.progress import ProgressCallback

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolishTelemetry:
    """Compact per-candidate polish telemetry (P12.5-E).

    Purely diagnostic: nothing here participates in ranking or persistence.
    """

    # Derivative provenance
    derivative_mode: str = DerivativeMode.REFERENCE_FD.value
    requested_mode: str = DerivativeMode.REFERENCE_FD.value
    fallback_reason: str | None = None

    # Cost
    wall_time_s: float = 0.0
    n_iterations: int = 0
    nfev: int = 0
    njev: int = 0
    nhev: int = 0
    constraint_nfev: int = 0
    constraint_njev: int = 0

    # Transaction / analytical work (zero in REFERENCE_FD)
    transaction_evaluations: int = 0
    transaction_reuse_hits: int = 0
    objective_jac_calls: int = 0
    constraint_jac_calls: int = 0
    jacobian_evals: int = 0
    factorizations: int = 0
    direct_substitutions: int = 0
    adjoint_substitutions: int = 0

    # P12.5-F nominal-work accounting (zero in REFERENCE_FD).
    #   transaction_nominal_sweep_solves  - frequencies the transaction had to
    #       assemble and screen itself (the former duplicate nominal sweep).
    #   transaction_nominal_states_reused - frequencies whose assembled/screened
    #       nominal state came from the production evaluator.
    transaction_nominal_sweep_solves: int = 0
    transaction_nominal_states_reused: int = 0
    nominal_bundle_hits: int = 0
    nominal_bundle_misses: int = 0
    nominal_states_captured: int = 0
    nominal_bundles_published: int = 0
    nominal_bundles_dropped: int = 0
    nominal_peak_retained_states: int = 0

    # Nominal MNA work performed through the production evaluator
    evaluator_unique_evaluations: int = 0
    evaluator_target_freq_solves: int = 0
    evaluator_coarse_freq_solves: int = 0

    # Outcome
    success: bool = False
    status_message: str = ""
    final_objective: float = float("nan")
    final_v_max: float = float("nan")
    final_feasible: bool = False

    # Problem shape (for A/B tables)
    n_params: int = 0
    n_constraint_rows: int = 0
    n_evaluation_frequencies: int = 0

    extra: dict[str, Any] = field(default_factory=dict)


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
    retained: EvaluationResult  # the better of pre/post
    method_used: str
    success: bool
    n_iterations: int
    n_evaluations: int
    termination: str
    reason: str | None  # why we fell back or failed
    telemetry: PolishTelemetry = field(default_factory=PolishTelemetry)


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
            import importlib.util

            if importlib.util.find_spec("cyipopt") is not None:
                return "ipopt"
            else:
                raise ImportError
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


@dataclass
class _PolishRun:
    """Internal outcome of one ``minimize`` attempt."""

    post_result: EvaluationResult
    success: bool
    n_iter: int
    term_msg: str
    reason: str | None
    n_evals: int
    telemetry: PolishTelemetry


def _run_polish(
    pre: EvaluationResult,
    context: EvaluationContext,
    cache: DomainEvaluatorCache,
    opt_spec: OptimizationSpec,
    method_str: str,
    mode: DerivativeMode,
    requested_mode: DerivativeMode,
    fallback_reason: str | None,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> _PolishRun:
    """Run one ``trust-constr`` polish under a single derivative mode.

    ``REFERENCE_FD`` reproduces the frozen P05 call exactly: ``jac="2-point"``
    on the objective and a ``NonlinearConstraint`` with no ``jac``.
    ``ANALYTICAL`` differs only in those two arguments.
    """
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    import foster_eom.optimize.perf as _perf

    domain_id = context.domain.domain_id
    _p = _perf.get_perf_stats()

    x0 = np.array(pre.x, dtype=np.float64)
    bounds = Bounds(lb=0.0, ub=1.0)
    fd_step = opt_spec.finite_difference_step
    max_iter = opt_spec.local_max_iterations

    n_evals_before = cache.n_unique_evaluations
    target_solves_before = cache.target_frequency_point_solves
    coarse_solves_before = cache.coarse_frequency_point_solves

    n_constraint_calls = 0

    def _obj(x: np.ndarray) -> float:
        if _p:
            _p.current_callback = "objective"
            _p.polish_evals += 1
            _p.record_x_eval(domain_id, tuple(x))
        return evaluate(x, context, cache).objective_value

    def _g_vec(x: np.ndarray) -> np.ndarray:
        nonlocal n_constraint_calls
        n_constraint_calls += 1
        if _p:
            _p.current_callback = "constraint"
            _p.polish_evals += 1
            _p.record_x_eval(domain_id, tuple(x))
        r = evaluate(x, context, cache)
        if not r.hard_margins:
            return np.array([1.0])
        return np.array(r.hard_margins, dtype=np.float64)

    provider: AnalyticalDerivativeProvider | None = None
    if mode == DerivativeMode.ANALYTICAL:
        provider = AnalyticalDerivativeProvider(context, cache)
        try:
            provider.preflight(x0)
        except DerivativeUnavailable:
            provider.release()
            raise
        obj_jac: Any = provider.objective_jac
        nlc = NonlinearConstraint(_g_vec, lb=0.0, ub=np.inf, jac=provider.constraint_jac)
    else:
        obj_jac = "2-point"
        nlc = NonlinearConstraint(_g_vec, lb=0.0, ub=np.inf)

    t_run = time.perf_counter()
    scipy_result: Any = None
    cancelled = False
    try:
        def _tc_callback(xk: Any, state: Any) -> bool:
            """Native trust-constr callback for cancellation and progress."""
            nonlocal cancelled
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                return True
            return False

        scipy_result = minimize(
            fun=_obj,
            x0=x0,
            method=method_str,
            bounds=bounds,
            constraints=nlc,
            jac=obj_jac,
            callback=_tc_callback,
            options={
                "maxiter": max_iter,
                "finite_diff_rel_step": fd_step,
                "verbose": 0,
            },
        )
        post_x = np.clip(scipy_result.x, 0.0, 1.0)
        post_result = evaluate(post_x, context, cache)
        success = not cancelled and bool(scipy_result.success)
        n_iter = getattr(scipy_result, "nit", 0)
        term_msg = "cancelled" if cancelled else scipy_result.message
        reason = None
    except DerivativeUnavailable:
        # Derivative problem, not a numerical polish failure — let the caller
        # fall this candidate back to REFERENCE_FD.  Release the shared nominal
        # state first so the FD reference run neither captures nor sees it.
        if provider is not None:
            provider.release()
        raise
    except Exception as exc:
        # Polish failed — retain pre-polish
        post_result = pre
        success = False
        n_iter = 0
        term_msg = f"exception: {type(exc).__name__}"
        reason = str(exc)
    dt_run = time.perf_counter() - t_run

    n_evals = cache.n_unique_evaluations - n_evals_before

    prov_metrics = (
        provider.metrics_snapshot()
        if provider is not None
        else {
            "transaction_evaluations": 0,
            "transaction_reuse_hits": 0,
            "objective_jac_calls": 0,
            "constraint_jac_calls": 0,
            "jacobian_evals": 0,
            "factorizations": 0,
            "direct_substitutions": 0,
            "adjoint_substitutions": 0,
            "transaction_nominal_sweep_solves": 0,
            "transaction_nominal_states_reused": 0,
            "nominal_bundle_hits": 0,
            "nominal_bundle_misses": 0,
            "nominal_states_captured": 0,
            "nominal_bundles_published": 0,
            "nominal_bundles_dropped": 0,
            "nominal_peak_retained_states": 0,
        }
    )
    if provider is not None:
        # Heavy nominal state is scoped to this polish run only.
        provider.release()

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    constr_nfev = 0
    constr_njev = 0
    if scipy_result is not None:
        constr_nfev = sum(_as_int(v) for v in getattr(scipy_result, "constr_nfev", []) or [])
        constr_njev = sum(_as_int(v) for v in getattr(scipy_result, "constr_njev", []) or [])

    telemetry = PolishTelemetry(
        derivative_mode=mode.value,
        requested_mode=requested_mode.value,
        fallback_reason=fallback_reason,
        wall_time_s=dt_run,
        n_iterations=_as_int(n_iter),
        nfev=_as_int(getattr(scipy_result, "nfev", 0)) if scipy_result is not None else 0,
        njev=_as_int(getattr(scipy_result, "njev", 0)) if scipy_result is not None else 0,
        nhev=_as_int(getattr(scipy_result, "nhev", 0)) if scipy_result is not None else 0,
        constraint_nfev=constr_nfev or n_constraint_calls,
        constraint_njev=constr_njev,
        transaction_evaluations=prov_metrics["transaction_evaluations"],
        transaction_reuse_hits=prov_metrics["transaction_reuse_hits"],
        objective_jac_calls=prov_metrics["objective_jac_calls"],
        constraint_jac_calls=prov_metrics["constraint_jac_calls"],
        jacobian_evals=prov_metrics["jacobian_evals"],
        factorizations=prov_metrics["factorizations"],
        direct_substitutions=prov_metrics["direct_substitutions"],
        adjoint_substitutions=prov_metrics["adjoint_substitutions"],
        transaction_nominal_sweep_solves=prov_metrics.get("transaction_nominal_sweep_solves", 0),
        transaction_nominal_states_reused=prov_metrics.get("transaction_nominal_states_reused", 0),
        nominal_bundle_hits=prov_metrics.get("nominal_bundle_hits", 0),
        nominal_bundle_misses=prov_metrics.get("nominal_bundle_misses", 0),
        nominal_states_captured=prov_metrics.get("nominal_states_captured", 0),
        nominal_bundles_published=prov_metrics.get("nominal_bundles_published", 0),
        nominal_bundles_dropped=prov_metrics.get("nominal_bundles_dropped", 0),
        nominal_peak_retained_states=prov_metrics.get("nominal_peak_retained_states", 0),
        evaluator_unique_evaluations=_as_int(n_evals),
        evaluator_target_freq_solves=_as_int(
            cache.target_frequency_point_solves - target_solves_before
        ),
        evaluator_coarse_freq_solves=_as_int(
            cache.coarse_frequency_point_solves - coarse_solves_before
        ),
        success=success,
        status_message=str(term_msg),
        final_objective=post_result.objective_value,
        final_v_max=post_result.v_max,
        final_feasible=post_result.feasible,
        n_params=_as_int(context.domain.variable_mapper.dimension),
        n_constraint_rows=max(_as_int(context.hard_layout.n), 1),
        n_evaluation_frequencies=_as_int(len(context.evaluation_frequencies_hz)),
    )

    return _PolishRun(
        post_result=post_result,
        success=success,
        n_iter=_as_int(n_iter),
        term_msg=str(term_msg),
        reason=reason,
        n_evals=n_evals,
        telemetry=telemetry,
    )


def polish_basin(
    basin: Basin,
    basin_index: int,
    context: EvaluationContext,
    cache: DomainEvaluatorCache,
    opt_spec: OptimizationSpec,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PolishResult:
    """Polish the representative of one basin.

    Derivatives come from ``opt_spec.local_derivative_mode``.  Under
    ``ANALYTICAL``, any unsupported / nonsmooth / unresolved derivative state or
    construction failure falls this candidate back to ``REFERENCE_FD``.
    If the polished result is Deb-worse than the pre-polish result, the
    pre-polish result is retained.
    """
    pre = basin.representative
    domain_id = context.domain.domain_id
    n_dim = context.domain.dimension

    import foster_eom.optimize.perf as _perf

    _p = _perf.get_perf_stats()
    if _p:
        _p.current_phase = "polish"
        _p.current_domain = domain_id
        _p.current_basin = f"basin_{basin_index}"
        _p.record_memory(f"before_polish_basin_{basin_index}")
    t_start = time.perf_counter()

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
            telemetry=PolishTelemetry(
                derivative_mode=DerivativeMode.REFERENCE_FD.value,
                requested_mode=opt_spec.local_derivative_mode.value,
                status_message="zero_dimensional",
                success=True,
                final_objective=pre.objective_value,
                final_v_max=pre.v_max,
                final_feasible=pre.feasible,
            ),
        )

    method_str = _resolve_method(opt_spec)
    requested_mode = opt_spec.local_derivative_mode

    run: _PolishRun
    if requested_mode == DerivativeMode.ANALYTICAL:
        try:
            run = _run_polish(
                pre,
                context,
                cache,
                opt_spec,
                method_str,
                DerivativeMode.ANALYTICAL,
                requested_mode,
                None,
                cancel_event,
                progress_callback,
            )
        except DerivativeUnavailable as exc:
            run = _run_polish(
                pre,
                context,
                cache,
                opt_spec,
                method_str,
                DerivativeMode.REFERENCE_FD,
                requested_mode,
                exc.reason,
                cancel_event,
                progress_callback,
            )
    else:
        run = _run_polish(
            pre,
            context,
            cache,
            opt_spec,
            method_str,
            DerivativeMode.REFERENCE_FD,
            requested_mode,
            None,
            cancel_event,
            progress_callback,
        )

    if _p:
        dt = time.perf_counter() - t_start
        _p.basin_polish_time[f"basin_{basin_index}"] = dt
        _p.basin_nit[f"basin_{basin_index}"] = run.n_iter
        if run.success:
            _p.basin_njev[f"basin_{basin_index}"] = run.telemetry.njev
            _p.basin_nfev[f"basin_{basin_index}"] = run.telemetry.nfev
        _p.basin_status[f"basin_{basin_index}"] = run.term_msg
        _p.basin_success[f"basin_{basin_index}"] = run.success
        _p.basin_telemetry[f"basin_{basin_index}"] = run.telemetry
        _p.polish_iterations += run.n_iter
        _p.record_memory(f"after_polish_basin_{basin_index}")
        _p.current_basin = None

    # Pre-polish retention: if post is Deb-worse, discard
    retained = pre if deb_better(pre, run.post_result) else run.post_result

    return PolishResult(
        domain_id=domain_id,
        basin_index=basin_index,
        pre_polish=pre,
        post_polish=run.post_result,
        retained=retained,
        method_used=method_str,
        success=run.success,
        n_iterations=run.n_iter,
        n_evaluations=run.n_evals,
        termination=run.term_msg,
        reason=run.reason,
        telemetry=run.telemetry,
    )


# ---------------------------------------------------------------------------
# Polish top-K basins
# ---------------------------------------------------------------------------


def polish_top_k(
    basins: list[Basin],
    context: EvaluationContext,
    cache: DomainEvaluatorCache,
    opt_spec: OptimizationSpec,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[PolishResult]:
    """Polish the top-k basin representatives (by Deb key of representative)."""
    k = opt_spec.polish_top_k
    sorted_basins = sorted(basins, key=lambda b: deb_key(b.representative))
    selected = sorted_basins[:k]

    results: list[PolishResult] = []
    for i, basin in enumerate(selected):
        # Cooperative cancellation between candidates.
        if cancel_event is not None and cancel_event.is_set():
            break
        pr = polish_basin(basin, i, context, cache, opt_spec, cancel_event, progress_callback)
        results.append(pr)

    return results
