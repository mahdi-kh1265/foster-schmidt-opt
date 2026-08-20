"""P12.5-F: Nominal MNA state reuse tests."""

import numpy as np

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.derivative_provider import AnalyticalDerivativeProvider
from foster_eom.optimize.evaluator import DomainEvaluatorCache, evaluate
from foster_eom.optimize.local_polish import polish_basin
from tests.unit.test_p12_5_e_analytical_polish import X0, _basin, _build_case, _spec


def test_f_same_u_reuse():
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    evaluate(X0, ctx, cache)

    provider.objective_jac(X0)
    provider.constraint_jac(X0)

    m = provider.metrics_snapshot()
    assert m["nominal_bundles_published"] == 1
    assert m["nominal_bundle_hits"] > 0
    assert m["transaction_nominal_sweep_solves"] == 0
    assert m["transaction_nominal_states_reused"] > 0


def test_f_new_u_invalidation():
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    evaluate(X0, ctx, cache)
    provider.objective_jac(X0)

    x1 = np.array([0.31, 0.62, 0.48])
    provider.objective_jac(x1)

    m = provider.metrics_snapshot()
    assert m["jacobian_evals"] == 2
    assert m["nominal_bundle_hits"] == 2
    assert m["nominal_bundle_misses"] == 0
    assert m["transaction_nominal_sweep_solves"] == 0


def test_f_context_invalidation():
    ctx1 = _build_case()
    ctx2 = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx1, cache)

    evaluate(X0, ctx1, cache)

    provider.context = ctx2
    provider.transaction.context = ctx2
    provider.objective_jac(X0)

    m = provider.metrics_snapshot()
    assert m["nominal_bundle_misses"] > 0


def test_f_failure_branch(monkeypatch):
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr("foster_eom.optimize.evaluator.solve_circuit_single_with_state", _boom)

    res = evaluate(X0, ctx, cache)
    assert res.numerical_status == "mna_singular"

    m = provider.metrics_snapshot()
    assert m["nominal_bundles_dropped"] == 1
    assert m["nominal_bundles_published"] == 0


def test_f_fd_isolation():
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.REFERENCE_FD))

    assert cache.nominal_exchange is None
    assert pr.telemetry.transaction_nominal_sweep_solves == 0


def test_f_derivative_parity():
    ctx = _build_case(feasible=True)

    cache_e = DomainEvaluatorCache()
    prov_e = AnalyticalDerivativeProvider(ctx, None)

    cache_f = DomainEvaluatorCache()
    prov_f = AnalyticalDerivativeProvider(ctx, cache_f)

    evaluate(X0, ctx, cache_f)
    evaluate(X0, ctx, cache_e)

    j_obj_e = prov_e.objective_jac(X0)
    j_con_e = prov_e.constraint_jac(X0)

    j_obj_f = prov_f.objective_jac(X0)
    j_con_f = prov_f.constraint_jac(X0)

    np.testing.assert_array_equal(j_obj_e, j_obj_f)
    np.testing.assert_array_equal(j_con_e, j_con_f)


def test_f_bounded_lifetime():
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    x1 = np.array([0.1, 0.2, 0.3])
    x2 = np.array([0.4, 0.5, 0.6])
    x3 = np.array([0.7, 0.8, 0.9])

    evaluate(x1, ctx, cache)
    evaluate(x2, ctx, cache)
    evaluate(x3, ctx, cache)

    m = provider.metrics_snapshot()
    assert m["nominal_retained_bundles"] <= 1
