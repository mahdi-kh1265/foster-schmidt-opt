"""P12.5-E: production ``trust-constr`` integration of the analytical Jacobians.

Proves, for the wired-in ANALYTICAL derivative mode:

  * the objective and constraint callbacks still return frozen production values;
  * the supplied Jacobians are exactly the validated transaction outputs;
  * one current-``u`` transaction is shared and reused, and a new ``u``
    invalidates it without retaining historical heavy state;
  * an unsupported / incomplete derivative state falls the candidate back to
    ``REFERENCE_FD`` rather than handing SciPy a partial Jacobian;
  * SciPy consumes the callables and performs no numerical differentiation;
  * the frozen ``REFERENCE_FD`` call is byte-identical to P05.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.objectives import DerivativeMode, LocalMethod, OptimizationSpec
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.dedup import Basin
from foster_eom.optimize.derivative_provider import (
    AnalyticalDerivativeProvider,
    DerivativeUnavailable,
)
from foster_eom.optimize.domain import ContinuousOptimizationDomain
from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    build_evaluation_context,
    evaluate,
)
from foster_eom.optimize.local_polish import polish_basin
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.sensitivities.objective_gradient import DerivativeStatus
from foster_eom.sensitivities.transaction import DerivativeTransaction
from tests.unit.test_sens_e2e import DummyEOM


class MatchableEOM(OnePortModel):
    """EOM load the single-cell test network can actually move |Gamma| against.

    ``DummyEOM`` (1 pF) is ~159 kOhm reactive at 1 MHz, which pins |Gamma| at 1
    and every match constraint hard-violated.  This load keeps the small case
    feasible with a real objective gradient, so the FD-vs-analytical smoke
    compares two runs that are actually solving something.
    """

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        return 120.0 - 1j / (2 * np.pi * np.asarray(f_hz) * 2e-10)

    @property
    def metadata(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Deterministic small case (real domain, real evaluation context)
# ---------------------------------------------------------------------------


def _build_case(base_grid_points: int = 6, off_target_v: float = 2.0, feasible: bool = False):
    """Build a real 3-parameter single-cell domain + evaluation context.

    Uses the established construction pattern of ``tests/unit/test_sens_e2e.py``.
    With ``base_grid_points > 1`` and a finite ``off_target_eom_peak_rms_v`` the
    hard layout carries off-target rows, so the analytical path exercises both
    the direct (target) and adjoint (off-target) routes.

    ``feasible=True`` widens the component/match windows and uses
    :class:`MatchableEOM` so the start point is feasible — required for a
    meaningful scientific-equivalence comparison.
    """
    topo = TopologyCandidate(
        branch1_cells=1,
        branch2_cells=0,
        branch1_has_c0=True,
        branch1_has_linf=False,
        branch2_has_c0=False,
        branch2_has_linf=False,
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        branch1_n_coefficients=2,
        branch2_n_coefficients=0,
        n_reactive=1,
        structurally_valid=True,
        prune_reason="",
    )
    sp = SignPattern(
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        signs=(1,),
        series_targets=(),
        shunt_targets=(),
        branch1_required_intervals=(),
        branch2_required_intervals=(),
        branch1_realization=BranchRealization.FINITE_FOSTER,
        branch2_realization=BranchRealization.ZERO_IMPEDANCE,
    )
    domain = ContinuousOptimizationDomain(
        domain_id="p12_5_e_small",
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        topology=topo,
        branch1_realization=BranchRealization.FINITE_FOSTER,
        branch2_realization=BranchRealization.ZERO_IMPEDANCE,
        seed_indices=(0,),
        pole_regions_branch1=((1e6, 10e6),),
        pole_regions_branch2=(),
        k_box_bounds_branch1=((1e9, 1e12),),
        k_box_bounds_branch2=(),
        k0_bounds_b1=(1e9, 1e12),
        k0_bounds_b2=None,
        k_inf_bounds_b1=None,
        k_inf_bounds_b2=None,
        n_movable_poles_branch1=1,
        n_movable_poles_branch2=0,
        variable_mapper=build_variable_mapper(
            branch1_n_cells=1,
            branch1_has_c0=True,
            branch1_has_linf=False,
            branch1_pole_regions=((1e6, 10e6),),
            branch1_k_box_bounds=((1e9, 1e12),),
            branch1_k0_bounds=(1e9, 1e12),
            branch1_kinf_bounds=None,
            branch1_fixed_k0=None,
            branch1_fixed_kinf=None,
            branch1_fixed_k_residues=(None,),
            branch1_fixed_f_poles_hz=(None,),
            branch2_n_cells=0,
            branch2_has_c0=False,
            branch2_has_linf=False,
            branch2_pole_regions=(),
            branch2_k_box_bounds=(),
            branch2_k0_bounds=None,
            branch2_kinf_bounds=None,
            branch2_fixed_k0=None,
            branch2_fixed_kinf=None,
            branch2_fixed_k_residues=(),
            branch2_fixed_f_poles_hz=(),
        ),
        dimension=3,
        structurally_feasible=True,
        infeasibility_reason=None,
        canonical_sign_pattern=sp,
    )
    source = SourceSpec(
        mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=50.0, z_ref_ohm=50.0
    )
    if feasible:
        limits = ContinuousLimits(
            l_min_h=1e-9, l_max_h=1e-3, c_min_f=1e-12, c_max_f=1e-6, i_max_a=1.0, v_max_v=100.0
        )
        match_c = MatchConstraints(
            gamma_max=1.0,
            resistance_min_ohm=1.0,
            resistance_max_ohm=5000.0,
            max_abs_reactance_ohm=5000.0,
        )
        eom: OnePortModel = MatchableEOM()
    else:
        limits = ContinuousLimits(
            l_min_h=1e-9, l_max_h=1e-6, c_min_f=1e-12, c_max_f=1e-9, i_max_a=1.0, v_max_v=100.0
        )
        match_c = MatchConstraints(gamma_max=0.5, resistance_max_ohm=50.0)
        eom = DummyEOM()
    obj = ObjectiveConfig(
        z_ref_ohm=50.0,
        w_gamma=1.0,
        w_voltage=0.0,
        w_loss=0.0,
        w_complexity=0.0,
        voltage_targets_rms_v=(),
        voltage_target_weights=(),
    )
    ctx = build_evaluation_context(
        domain=domain,
        source_spec=source,
        eom_model=eom,
        component_limits=limits,
        match_constraints=match_c,
        stress_constraints=StressConstraints(
            source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=off_target_v
        ),
        target_frequencies_hz=(1.0e6,),
        sweep_f_min_hz=1.0e6,
        sweep_f_max_hz=2.0e6,
        base_grid_points=base_grid_points,
        objective_config=obj,
        feasibility_tolerance=1e-3,
        near_feasibility_tolerance=1e-3,
    )
    return ctx


def _spec(mode: DerivativeMode, max_iter: int = 12) -> OptimizationSpec:
    """Optimization spec that differs ONLY in derivative mode."""
    return OptimizationSpec(
        local_method=LocalMethod.TRUST_CONSTR,
        local_fallback_method=LocalMethod.TRUST_CONSTR,
        local_max_iterations=max_iter,
        polish_top_k=1,
        local_derivative_mode=mode,
    )


X0 = np.array([0.5, 0.5, 0.5])


def _basin(ctx, cache, x0=None):
    res = evaluate(X0 if x0 is None else x0, ctx, cache)
    return Basin(representative=res, members=[res])


# ---------------------------------------------------------------------------
# 1. Callbacks still return frozen production values
# ---------------------------------------------------------------------------


def test_analytical_mode_leaves_objective_and_constraint_values_frozen():
    """ANALYTICAL changes only ``jac``; values come from the frozen evaluator."""
    ctx = _build_case()
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)
    with patch("scipy.optimize.minimize", side_effect=_capture):
        polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    fun = captured["fun"]
    nlc = captured["constraints"]

    for x in (X0, np.array([0.3, 0.7, 0.45]), np.array([0.9, 0.1, 0.8])):
        ref = evaluate(x, ctx, cache)
        assert fun(x) == ref.objective_value
        np.testing.assert_array_equal(nlc.fun(x), np.array(ref.hard_margins, dtype=np.float64))


def test_constraint_jacobian_row_count_matches_margin_count():
    """Constraint order/shape parity between values and Jacobian."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    margins = evaluate(X0, ctx, cache).hard_margins
    provider = AnalyticalDerivativeProvider(ctx)
    j_g = provider.constraint_jac(X0)
    assert j_g.shape == (len(margins), ctx.domain.variable_mapper.dimension)
    assert ctx.hard_layout.n == len(margins)


# ---------------------------------------------------------------------------
# 2. Jacobians are exactly the validated transaction outputs
# ---------------------------------------------------------------------------


def test_provider_jacobians_equal_transaction_outputs():
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)
    reference = DerivativeTransaction(ctx)

    for x in (X0, np.array([0.42, 0.61, 0.55])):
        j_base_ref, j_constr_ref = reference.evaluate_jacobians(x)
        np.testing.assert_array_equal(provider.objective_jac(x), j_base_ref)
        np.testing.assert_array_equal(provider.constraint_jac(x), j_constr_ref)


# ---------------------------------------------------------------------------
# 3. Same-u reuse / new-u invalidation
# ---------------------------------------------------------------------------


def test_same_u_reuses_one_transaction():
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)

    provider.objective_jac(X0)
    provider.constraint_jac(X0)
    provider.objective_jac(X0.copy())

    assert provider.transaction.metrics["jacobian_evals"] == 1
    assert provider.n_transaction_evaluations == 1
    assert provider.n_reuse_hits == 2


def test_new_u_invalidates_and_retains_no_historical_state():
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)
    txn = provider.transaction

    x1 = X0
    x2 = np.array([0.31, 0.62, 0.48])
    provider.objective_jac(x1)
    provider.constraint_jac(x1)
    provider.objective_jac(x2)
    provider.constraint_jac(x2)

    assert txn.metrics["jacobian_evals"] == 2
    assert provider.n_transaction_evaluations == 2
    # Only the current u's heavy state is held: single slot, latest u.
    np.testing.assert_array_equal(txn.current_x, x2)
    assert txn._j_base is not None and txn._j_constr is not None
    # Re-asking for x1 must rebuild, proving nothing historical was retained.
    provider.objective_jac(x1)
    assert txn.metrics["jacobian_evals"] == 3
    np.testing.assert_array_equal(txn.current_x, x1)


# ---------------------------------------------------------------------------
# 4. Fallback on unsupported / incomplete derivative state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_status",
    [
        DerivativeStatus.UNSUPPORTED,
        DerivativeStatus.NONSMOOTH_KINK,
        DerivativeStatus.NUMERICALLY_UNRESOLVED,
        DerivativeStatus.NOMINAL_FAILURE,
    ],
)
def test_non_smooth_status_raises_derivative_unavailable(monkeypatch, bad_status):
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)
    provider.objective_jac(X0)  # build once so last_result exists

    monkeypatch.setattr(
        type(provider.transaction),
        "last_status",
        property(lambda _self: bad_status),
    )
    with pytest.raises(DerivativeUnavailable) as exc:
        provider.objective_jac(np.array([0.4, 0.4, 0.4]))
    assert bad_status.value in exc.value.reason


def test_incomplete_nominal_coverage_raises(monkeypatch):
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)
    monkeypatch.setattr(
        type(provider.transaction),
        "solved_target_indices",
        property(lambda _self: frozenset()),
    )
    with pytest.raises(DerivativeUnavailable) as exc:
        provider.objective_jac(X0)
    assert "nominal_target_solve_failed" in exc.value.reason


def test_unsupported_state_falls_back_to_reference_fd(monkeypatch):
    """A bad derivative state must polish the candidate under FD instead."""
    ctx = _build_case()

    def _boom(_self, _x):
        raise RuntimeError("synthetic derivative construction failure")

    monkeypatch.setattr(DerivativeTransaction, "evaluate_jacobians", _boom)

    cache_a = DomainEvaluatorCache()
    pr = polish_basin(_basin(ctx, cache_a), 0, ctx, cache_a, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.requested_mode == DerivativeMode.ANALYTICAL.value
    assert pr.telemetry.derivative_mode == DerivativeMode.REFERENCE_FD.value
    assert pr.telemetry.fallback_reason is not None
    assert "construction_failed" in pr.telemetry.fallback_reason
    # No analytical work was charged.
    assert pr.telemetry.factorizations == 0
    assert pr.telemetry.transaction_evaluations == 0

    # And the fallback reproduces the pure-FD run from the same start.
    cache_b = DomainEvaluatorCache()
    pr_fd = polish_basin(_basin(ctx, cache_b), 0, ctx, cache_b, _spec(DerivativeMode.REFERENCE_FD))
    np.testing.assert_allclose(pr.retained.x, pr_fd.retained.x, rtol=0, atol=0)
    assert pr.retained.objective_value == pr_fd.retained.objective_value


def test_fallback_never_emits_partial_jacobian(monkeypatch):
    """A non-finite entry must be rejected, not passed on."""
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)
    n = ctx.domain.variable_mapper.dimension

    def _nan_jac(_self, x):
        _self._invalidate_cache(np.asarray(x, dtype=np.float64))
        bad = np.full(n, np.nan)
        return bad, np.zeros((ctx.hard_layout.n, n))

    monkeypatch.setattr(DerivativeTransaction, "evaluate_jacobians", _nan_jac)
    with pytest.raises(DerivativeUnavailable):
        provider.objective_jac(X0)


# ---------------------------------------------------------------------------
# 5. SciPy really consumes the callables (multi-signal proof)
# ---------------------------------------------------------------------------


def test_analytical_supplies_callable_jacobians_and_fd_does_not():
    """Signal (b): captured jac arguments."""
    ctx = _build_case()
    captured: dict[str, dict] = {}

    def _capture(**kwargs):
        captured.clear()
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    for mode in (DerivativeMode.ANALYTICAL, DerivativeMode.REFERENCE_FD):
        cache = DomainEvaluatorCache()
        with patch("scipy.optimize.minimize", side_effect=_capture):
            polish_basin(_basin(ctx, cache), 0, ctx, cache, _spec(mode))
        jac = captured["jac"]
        c_jac = captured["constraints"].jac
        if mode == DerivativeMode.ANALYTICAL:
            assert callable(jac) and jac not in ("2-point", "3-point", "cs")
            assert callable(c_jac) and c_jac not in ("2-point", "3-point", "cs")
        else:
            assert jac == "2-point"
            assert c_jac == "2-point"  # SciPy's frozen NonlinearConstraint default


def test_analytical_triggers_no_scipy_numerical_differentiation():
    """Signal (a): ``approx_derivative`` must never fire under ANALYTICAL."""
    import scipy.optimize._differentiable_functions as sdf

    ctx = _build_case()
    counts = {}

    for mode in (DerivativeMode.REFERENCE_FD, DerivativeMode.ANALYTICAL):
        calls: list[int] = []

        def _spy(*args, _calls=calls, _real=sdf.approx_derivative, **kwargs):
            _calls.append(1)
            return _real(*args, **kwargs)

        cache = DomainEvaluatorCache()
        basin = _basin(ctx, cache)
        with patch.object(sdf, "approx_derivative", _spy):
            pr = polish_basin(basin, 0, ctx, cache, _spec(mode))
        assert pr.telemetry.derivative_mode == mode.value, "unexpected fallback"
        counts[mode] = (len(calls), pr.telemetry)

    fd_calls, fd_tel = counts[DerivativeMode.REFERENCE_FD]
    an_calls, an_tel = counts[DerivativeMode.ANALYTICAL]

    assert an_calls == 0, f"ANALYTICAL numerically differentiated {an_calls} times"
    assert fd_calls > 0, "REFERENCE_FD unexpectedly avoided numerical differentiation"

    # Signal (c): SciPy telemetry shows our Jacobians being consumed, and the
    # Np-fold function-evaluation pattern is gone.
    n_p = ctx.domain.variable_mapper.dimension
    assert an_tel.njev > 0
    assert an_tel.constraint_njev > 0
    assert an_tel.nfev <= an_tel.njev + 2, "objective still evaluated in an FD pattern"
    assert fd_tel.nfev >= fd_tel.njev * (n_p + 1) - n_p, "FD baseline lost its Np multiplier"


# ---------------------------------------------------------------------------
# 6. The frozen REFERENCE_FD call is unchanged
# ---------------------------------------------------------------------------


def test_reference_fd_minimize_call_is_frozen():
    ctx = _build_case()
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    spec = _spec(DerivativeMode.REFERENCE_FD, max_iter=1500)
    cache = DomainEvaluatorCache()
    with patch("scipy.optimize.minimize", side_effect=_capture):
        polish_basin(_basin(ctx, cache), 0, ctx, cache, spec)

    assert captured["method"] == "trust-constr"
    assert captured["jac"] == "2-point"
    assert captured["options"] == {
        "maxiter": 1500,
        "finite_diff_rel_step": spec.finite_difference_step,
        "verbose": 0,
    }
    assert captured["bounds"].lb == 0.0
    assert captured["bounds"].ub == 1.0
    nlc = captured["constraints"]
    assert nlc.lb == 0.0
    assert nlc.ub == np.inf
    assert nlc.jac == "2-point"
    assert set(captured) == {"fun", "x0", "method", "bounds", "constraints", "jac", "options"}


def test_analytical_preserves_every_other_minimize_argument():
    """Only ``jac`` (objective + constraint) may differ between the two modes."""
    ctx = _build_case()
    grabbed: dict[str, dict] = {}

    def _make(mode_key):
        def _capture(**kwargs):
            grabbed[mode_key] = kwargs
            raise RuntimeError("stop after capture")

        return _capture

    for key, mode in (("fd", DerivativeMode.REFERENCE_FD), ("an", DerivativeMode.ANALYTICAL)):
        cache = DomainEvaluatorCache()
        with patch("scipy.optimize.minimize", side_effect=_make(key)):
            polish_basin(_basin(ctx, cache), 0, ctx, cache, _spec(mode))

    fd, an = grabbed["fd"], grabbed["an"]
    assert set(fd) == set(an)
    assert fd["method"] == an["method"]
    assert fd["options"] == an["options"]
    np.testing.assert_array_equal(fd["x0"], an["x0"])
    assert (fd["bounds"].lb, fd["bounds"].ub) == (an["bounds"].lb, an["bounds"].ub)
    assert (fd["constraints"].lb, fd["constraints"].ub) == (
        an["constraints"].lb,
        an["constraints"].ub,
    )
    assert fd["constraints"].keep_feasible == an["constraints"].keep_feasible


# ---------------------------------------------------------------------------
# 7. Empty-layout parity and spec plumbing
# ---------------------------------------------------------------------------


def test_empty_hard_layout_jacobian_matches_placeholder_row():
    ctx = _build_case()
    provider = AnalyticalDerivativeProvider(ctx)
    provider.n_hard = 0
    provider.n_constraint_rows = 1
    j_g = provider.constraint_jac(X0)
    assert j_g.shape == (1, ctx.domain.variable_mapper.dimension)
    np.testing.assert_array_equal(j_g, np.zeros_like(j_g))


def test_derivative_mode_defaults_to_reference_fd():
    assert OptimizationSpec().local_derivative_mode == DerivativeMode.REFERENCE_FD


def test_derivative_mode_round_trips_through_yaml():
    import yaml

    from foster_eom.persistence.yaml_io import _dict_to_spec, _spec_to_dict

    base = _spec_to_dict.__module__  # keep import used if signature changes
    assert base

    path = "fs-theo/examples/design_spec.example.yaml"
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    spec = _dict_to_spec(data)
    assert spec.optimization.local_derivative_mode == DerivativeMode.REFERENCE_FD

    data["optimization"]["local"]["derivative_mode"] = "analytical"
    spec2 = _dict_to_spec(data)
    assert spec2.optimization.local_derivative_mode == DerivativeMode.ANALYTICAL
    out = _spec_to_dict(spec2)
    assert out["optimization"]["local"]["derivative_mode"] == "analytical"


def test_zero_dimensional_polish_still_short_circuits():
    ctx = _build_case()
    mock_ctx = MagicMock()
    mock_ctx.domain.dimension = 0
    mock_ctx.domain.domain_id = "zero"
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)
    pr = polish_basin(basin, 0, mock_ctx, cache, _spec(DerivativeMode.ANALYTICAL))
    assert pr.termination == "zero_dimensional"
    assert pr.method_used == "none"
    assert pr.retained is basin.representative


# ---------------------------------------------------------------------------
# 8. Deterministic FD-vs-analytical smoke from the same start
# ---------------------------------------------------------------------------


def test_fd_vs_analytical_polish_smoke_from_same_start():
    """Small deterministic A/B. Sanity only — no performance conclusions."""
    ctx = _build_case(feasible=True)

    cache_fd = DomainEvaluatorCache()
    pr_fd = polish_basin(
        _basin(ctx, cache_fd), 0, ctx, cache_fd, _spec(DerivativeMode.REFERENCE_FD)
    )
    cache_an = DomainEvaluatorCache()
    pr_an = polish_basin(_basin(ctx, cache_an), 0, ctx, cache_an, _spec(DerivativeMode.ANALYTICAL))

    assert pr_an.telemetry.derivative_mode == DerivativeMode.ANALYTICAL.value
    assert pr_an.telemetry.fallback_reason is None
    np.testing.assert_array_equal(pr_fd.pre_polish.x, pr_an.pre_polish.x)

    # Scientific equivalence of the retained candidate.
    assert pr_fd.retained.feasible == pr_an.retained.feasible
    assert pr_an.retained.v_max <= max(1e-6, pr_fd.retained.v_max * (1 + 1e-3)) + 1e-9
    d_obj = abs(pr_an.retained.objective_value - pr_fd.retained.objective_value)
    scale = max(abs(pr_fd.retained.objective_value), 1e-12)
    assert d_obj / scale <= 1e-3 or d_obj <= 1e-6

    # Analytical really did the analytical work.
    assert pr_an.telemetry.factorizations > 0
    assert pr_an.telemetry.direct_substitutions > 0
    assert pr_an.telemetry.transaction_evaluations > 0
