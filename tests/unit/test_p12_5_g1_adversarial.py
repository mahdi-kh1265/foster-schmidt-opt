"""P12.5-G1: Adversarial Nominal-State Identity / Invalidation Audit."""

import numpy as np

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.derivative_provider import AnalyticalDerivativeProvider
from foster_eom.optimize.evaluator import DomainEvaluatorCache, canonical_x, evaluate
from foster_eom.optimize.local_polish import polish_basin
from tests.unit.test_p12_5_e_analytical_polish import X0, _basin, _build_case, _spec


def test_g1_exact_repeated_coordinate():
    """G1-A: Exact repeated coordinate must hit."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    evaluate(X0, ctx, cache)
    provider.objective_jac(X0)

    # Repeat the same request
    provider.objective_jac(X0)
    provider.constraint_jac(X0)

    m = provider.metrics_snapshot()
    assert m["nominal_bundle_hits"] == 1
    assert m["nominal_bundle_misses"] == 0
    # No new sweeps for transaction
    assert m["transaction_nominal_sweep_solves"] == 0


def test_g1_changed_coordinate_miss():
    """G1-B: Changed coordinate must miss (1 ULP or small change)."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    evaluate(X0, ctx, cache)
    provider.objective_jac(X0)

    m1 = provider.metrics_snapshot()
    assert m1["nominal_bundle_hits"] == 1
    assert m1["nominal_bundle_misses"] == 0

    # 1. Completely different coordinate
    x1 = np.array([0.31, 0.62, 0.48])
    # The _ensure() logic inside provider will call evaluate() and overwrite the bundle.
    # To test a pure MISS of the transaction cache, we must call transaction directly
    # OR change the provider's logic. But wait, `provider.objective_jac(x)` calls evaluate
    # first, which populates the cache for the NEW x, so evaluate_jacobians gets a hit!
    # So to test a pure MISS on the exchange, we can use the exchange explicitly:
    exchange = cache.nominal_exchange
    assert exchange is not None

    _, x1_key = canonical_x(x1)
    bundle_x1 = exchange.lookup(x1_key, ctx)
    assert bundle_x1 is None
    assert exchange.lookup_misses == 1

    # 2. 1 ULP float change
    x_ulp = X0.copy()
    x_ulp[0] = np.nextafter(x_ulp[0], 1.0)
    _, x_ulp_key = canonical_x(x_ulp)

    bundle_ulp = exchange.lookup(x_ulp_key, ctx)
    assert bundle_ulp is None
    assert exchange.lookup_misses == 2

    # 3. Request evaluation of the ULP change
    provider.objective_jac(x_ulp)
    m2 = provider.metrics_snapshot()
    # Evaluating x_ulp caused a hit because `_ensure` ran `evaluate` first
    assert m2["nominal_bundle_hits"] == 2


def test_g1_clipping_equivalence():
    """G1-C: Production clipping equivalence."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    # Raw coordinate slightly outside bounds
    raw_u = np.array([-1e-15, 1.000000000000001, 0.5])
    # Evaluate populates the cache using the canonicalized (clipped) coordinate
    evaluate(raw_u, ctx, cache)

    # The transaction should be able to hit the cache using raw_u
    # because `_ensure` uses `canonical_x(raw_u)` and `transaction` does too.
    provider.objective_jac(raw_u)

    m = provider.metrics_snapshot()
    assert m["nominal_bundle_hits"] == 1
    assert m["nominal_bundle_misses"] == 0


def test_g1_context_invalidation():
    """G1-D: Similar-looking but distinct contexts must miss."""
    ctx1 = _build_case()
    ctx2 = _build_case()  # distinct object, identical data
    cache = DomainEvaluatorCache()
    AnalyticalDerivativeProvider(ctx1, cache)

    evaluate(X0, ctx1, cache)

    exchange = cache.nominal_exchange
    assert exchange is not None

    _, x_key = canonical_x(X0)

    # Must hit with original context
    assert exchange.lookup(x_key, ctx1) is not None

    # Must miss with new context (object identity check)
    assert exchange.lookup(x_key, ctx2) is None


def test_g1_frequency_grid_mutation():
    """G1-E: Frequency-grid mutation."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    AnalyticalDerivativeProvider(ctx, cache)

    evaluate(X0, ctx, cache)
    exchange = cache.nominal_exchange
    _, x_key = canonical_x(X0)

    # Mutate the frequencies in a fake context to prove identity rejection
    import dataclasses
    ctx_mutated = dataclasses.replace(ctx, evaluation_frequencies_hz=(1e6, 2e6))

    assert exchange.lookup(x_key, ctx_mutated) is None


def test_g1_failed_state_isolation(monkeypatch):
    """G1-F: Failed / nonfinite nominal state."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    AnalyticalDerivativeProvider(ctx, cache)

    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr("foster_eom.optimize.evaluator.solve_circuit_single_with_state", _boom)

    evaluate(X0, ctx, cache)

    exchange = cache.nominal_exchange
    assert exchange is not None
    # Failed evaluation must drop the bundle, not publish it
    assert exchange.bundles_dropped == 1
    assert exchange.bundles_published == 0

    _, x_key = canonical_x(X0)
    assert exchange.lookup(x_key, ctx) is None


def test_g1_bounded_lifetime():
    """G1-G: State replacement and bounded lifetime."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    provider = AnalyticalDerivativeProvider(ctx, cache)

    n_iter = 15
    for _i in range(n_iter):
        x = np.random.rand(len(X0))
        evaluate(x, ctx, cache)
        provider.objective_jac(x)

    m = provider.metrics_snapshot()
    # At most 1 committed bundle
    assert m["nominal_retained_bundles"] <= 1
    # Hits should equal n_iter (each _ensure calls evaluate which populates cache)
    assert m["nominal_bundle_hits"] == n_iter


def test_g1_fd_isolation():
    """G1-H: REFERENCE_FD isolation."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.REFERENCE_FD))

    # FD should not touch exchange
    assert cache.nominal_exchange is None
    assert pr.telemetry.transaction_nominal_sweep_solves == 0


def test_g1_real_callback_reuse():
    """G1-I: Real callback regression."""
    ctx = _build_case(feasible=True)
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    # Polish with analytical Jacobians
    pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    # The first Jacobian eval (preflight) will miss because X0 is already in DomainEvaluatorCache
    # so evaluate() skips publication. The subsequent 9 iterations will hit.
    assert pr.telemetry.nominal_bundle_hits > 0
    assert pr.telemetry.nominal_bundle_misses == 1
    assert pr.telemetry.transaction_nominal_sweep_solves > 0
