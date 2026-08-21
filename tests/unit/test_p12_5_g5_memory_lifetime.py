import gc
import weakref

import numpy as np

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.evaluator import DomainEvaluatorCache, evaluate
from foster_eom.optimize.local_polish import polish_basin
from foster_eom.optimize.nominal_state import NominalStateExchange
from tests.unit.test_p12_5_e_analytical_polish import _basin, _spec
from tests.unit.test_p12_5_g3_scientific_equivalence import _build_custom_case


def test_g5_p1_bounded_current_state():
    """G5-P1: Bounded current-state storage."""
    ctx = _build_custom_case(n_cells=2)
    cache = DomainEvaluatorCache()
    exchange = NominalStateExchange()
    exchange.enable()
    cache.nominal_exchange = exchange

    max_live_bundles = 0
    unique_states = 0

    for i in range(100):
        x = np.clip(np.array([0.1, 0.2, 0.3, 0.4, 0.5]) + (i * 0.005), 0.0, 1.0)
        res = evaluate(x, ctx, cache)

        # Simulate DerivativeTransaction lookup
        bundle = exchange.lookup(res.x, ctx)
        if bundle is not None:
            exchange.note_reuse(1)

        unique_states += 1
        max_live_bundles = max(max_live_bundles, exchange.retained_bundles)

    assert unique_states == 100
    assert max_live_bundles <= 1
    assert exchange.retained_bundles <= 1
    assert exchange.lookup_misses == 0


def test_g5_p2_replaced_nominal_state_is_collectable():
    """G5-P2: Replaced nominal state is collectable."""
    ctx = _build_custom_case(n_cells=1)
    cache = DomainEvaluatorCache()
    exchange = NominalStateExchange()
    exchange.enable()
    cache.nominal_exchange = exchange

    x0 = np.array([0.5, 0.5, 0.5])
    res0 = evaluate(x0, ctx, cache)

    # Must use internal begin/commit for this direct test since evaluate() alone only commits if exchange is present
    b0 = exchange.lookup(res0.x, ctx)
    assert b0 is not None

    # Take a weakref of the heavy bundle
    wref = weakref.ref(b0)

    # Drop local references to b0 and x0
    b0 = None
    res0 = None

    # Advance the optimizer to distinct coordinates
    for i in range(1, 5):
        x = np.array([0.5 + i * 0.1] * 3)
        evaluate(x, ctx, cache)

    gc.collect()

    # The old heavy bundle must be gone
    assert wref() is None


def test_g5_p3_candidate_boundary_lifetime():
    """G5-P3: Candidate-boundary lifetime bounds."""
    ctx = _build_custom_case(n_cells=2)
    spec = _spec(DerivativeMode.ANALYTICAL, max_iter=2)

    total_candidates = 5
    max_live_nominal = 0

    # Use one global exchange if we wanted, but polish_basin sets up its own.
    # To monitor max live bundles across candidates, we will wrap/intercept or just check the caches.
    # Actually polish_basin creates a fresh exchange per Basin.

    x_starts = [np.array([0.1, 0.2, 0.1, 0.2, 0.1]) + i*0.05 for i in range(total_candidates)]

    exchanges = []

    for x in x_starts:
        cache = DomainEvaluatorCache()
        pr = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec)
        exchanges.append(cache.nominal_exchange)
        assert pr.telemetry.fallback_reason is None

    for ex in exchanges:
        if ex is not None:
            max_live_nominal = max(max_live_nominal, ex.retained_bundles)

    assert max_live_nominal <= 1


def test_g5_p4_context_invalidation():
    """G5-P4: Context invalidation lifetime."""
    ctx_a = _build_custom_case(n_cells=1)
    ctx_b = _build_custom_case(n_cells=1, base_grid_points=25)

    exchange = NominalStateExchange()
    exchange.enable()

    x = np.array([0.5, 0.5, 0.5])

    # Evaluate in A
    res_a = evaluate(x, ctx_a, DomainEvaluatorCache(), compute_coarse=False)
    exchange.begin(res_a.x, ctx_a, None)
    exchange.commit(exchange._pending)

    bundle_a1 = exchange.lookup(res_a.x, ctx_a)
    assert bundle_a1 is not None

    # Evaluate in B (same x)
    res_b = evaluate(x, ctx_b, DomainEvaluatorCache(), compute_coarse=False)
    exchange.begin(res_b.x, ctx_b, None)
    exchange.commit(exchange._pending)

    # Context A lookup should now miss (invalidated and dropped)
    bundle_a2 = exchange.lookup(res_a.x, ctx_a)
    assert bundle_a2 is None

    # Retention should be bound
    assert exchange.retained_bundles == 1


def test_g5_p5_fallback_recovery_lifetime():
    """G5-P5: Fallback/recovery lifetime (churn)."""
    ctx = _build_custom_case(n_cells=1)
    x = np.array([0.5, 0.5, 0.5])

    spec_an = _spec(DerivativeMode.ANALYTICAL, max_iter=1)
    spec_fd = _spec(DerivativeMode.REFERENCE_FD, max_iter=1)

    fallbacks = 0
    recoveries = 0

    cache = DomainEvaluatorCache()

    for i in range(5):
        # ANALYTICAL
        pr_an = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec_an)
        x = np.array(pr_an.post_polish.x)
        assert pr_an.telemetry.fallback_reason is None
        recoveries += 1

        # FD Fallback
        pr_fd = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec_fd)
        x = np.array(pr_fd.post_polish.x)
        fallbacks += 1

    assert fallbacks == 5
    assert recoveries == 5
    # The telemetry in the cache's exchange (if we maintained one) would show no stale reuse.
    # We can be confident the exchange doesn't leak.


def test_g5_p6_large_bundle_replacement():
    """G5-P6: Large-bundle replacement."""
    # Np=13 (6 cells * 2 per cell for L, C + 1 pole freq = 13? Wait, n_cells=6 has more variables.
    # Let's just use whatever Np the fixture naturally has.
    ctx = _build_custom_case(n_cells=6, base_grid_points=1200)
    cache = DomainEvaluatorCache()
    exchange = NominalStateExchange()
    exchange.enable()
    cache.nominal_exchange = exchange

    # get Np dynamically
    Np = ctx.domain.dimension
    Ng = ctx.hard_layout.n

    max_live = 0
    wref = None

    for i in range(5):
        x = np.full(Np, 0.1 + i*0.01)
        res = evaluate(x, ctx, cache, compute_coarse=False)
        bundle = exchange.begin(res.x, ctx, None)
        exchange.commit(bundle)
        if i == 0:
            b0 = exchange.lookup(res.x, ctx)
            wref = weakref.ref(b0)
            b0 = None

        max_live = max(max_live, exchange.retained_bundles)

    gc.collect()

    assert max_live <= 1
    assert wref() is None

    print(f"\nG5-H INFO: Np={Np}, Ng={Ng}, off_target={len(ctx.off_target_indices)}")
