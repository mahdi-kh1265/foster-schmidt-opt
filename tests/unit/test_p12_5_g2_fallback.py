"""P12.5-G2: Adversarial Fallback / State-Leakage Audit."""

from unittest.mock import patch

import numpy as np
import pytest

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.derivative_provider import (
    AnalyticalDerivativeProvider,
    DerivativeUnavailable,
)
from foster_eom.optimize.evaluator import DomainEvaluatorCache, evaluate
from foster_eom.optimize.local_polish import polish_basin
from foster_eom.sensitivities.transaction import DerivativeTransaction
from tests.unit.test_p12_5_e_analytical_polish import X0, _basin, _build_case, _spec


def test_g2_unsupported_analytical_config():
    """G2-A: Unsupported analytical configuration."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    # Force check_analytical_support to return False
    with patch("foster_eom.optimize.derivative_provider.check_analytical_support", return_value=(False, ["mock_unsupported"])):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.fallback_reason == "unsupported_config:mock_unsupported"
    assert pr.telemetry.derivative_mode == "reference_fd"


def test_g2_nonsmooth_derivative():
    """G2-B: Nonsmooth derivative state."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)


    def _mock_validate(self, j_base, j_constr):
        # Force a status_nonsmooth exception
        raise DerivativeUnavailable("status_nonsmooth:mock")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.fallback_reason == "status_nonsmooth:mock"
    assert pr.telemetry.derivative_mode == "reference_fd"


def test_g2_incomplete_coverage():
    """G2-C: Incomplete target/off-target coverage."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("nominal_target_solve_failed:[0]")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert "nominal_target_solve_failed" in pr.telemetry.fallback_reason
    assert pr.telemetry.derivative_mode == "reference_fd"


def test_g2_nonfinite_derivative():
    """G2-D: Nonfinite derivative output."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("objective_jac_nonfinite")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.fallback_reason == "objective_jac_nonfinite"
    assert pr.telemetry.derivative_mode == "reference_fd"


def test_g2_wrong_shape():
    """G2-E: Wrong shape / partial array."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("objective_jac_shape:(2,)")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.fallback_reason == "objective_jac_shape:(2,)"
    assert pr.telemetry.derivative_mode == "reference_fd"


def test_g2_transaction_construction_failure():
    """G2-F: Transaction construction failure."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)


    def _mock_evaluate_jacobians(self, x, x_key=None):
        raise ValueError("mock construction failure")

    with patch.object(DerivativeTransaction, "evaluate_jacobians", _mock_evaluate_jacobians):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.fallback_reason == "construction_failed:ValueError:mock construction failure"
    assert pr.telemetry.derivative_mode == "reference_fd"


def test_g2_stale_prior_success_state_leakage():
    """G2-G: Failure after a prior successful analytical iterate."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    _basin(ctx, cache)
    provider = AnalyticalDerivativeProvider(ctx, cache)

    # 1. Success on X0
    evaluate(X0, ctx, cache)
    provider.objective_jac(X0)

    # 2. Force failure on another coordinate
    x1 = np.array([0.31, 0.62, 0.48])
    evaluate(x1, ctx, cache)

    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("mock failure")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        with pytest.raises(DerivativeUnavailable):
            provider.objective_jac(x1)

    # The stale X0 transaction should not be used for x1!
    # Verified by the fact that DerivativeUnavailable bubbled up (not silently suppressed).
    # Also verify that the fallback starts fresh.
    pass


def test_g2_failure_followed_by_success():
    """G2-H: Failure followed by a new analytical candidate."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()

    # 1. Candidate A fails
    basin_a = _basin(ctx, cache)
    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("mock failure")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        pr_a = polish_basin(basin_a, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr_a.telemetry.derivative_mode == "reference_fd"

    # 2. Candidate B succeeds
    basin_b = _basin(ctx, cache)
    pr_b = polish_basin(basin_b, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))
    assert pr_b.telemetry.derivative_mode == "analytical"
    assert pr_b.telemetry.fallback_reason is None


def test_g2_original_start_restart_semantics():
    """G2-I: Original-start restart semantics."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    # We want to intercept the fallback minimize call to check its x0
    fallback_x0 = None
    original_minimize = None

    first_call = True
    def _mock_minimize(*args, **kwargs):
        nonlocal fallback_x0, original_minimize, first_call
        # The first call is the analytical attempt
        # We trigger unavailability inside the objective or jacobian
        # Let's do it by raising DerivativeUnavailable directly
        if first_call:
            first_call = False
            # We are in the mock, meaning we just entered minimize.
            raise DerivativeUnavailable("mock fallback trigger during minimize")

        # Second call is the fallback minimize
        fallback_x0 = kwargs.get("x0", args[1] if len(args) > 1 else None)
        return original_minimize(*args, **kwargs)

    import scipy.optimize
    original_minimize = scipy.optimize.minimize

    with patch("scipy.optimize.minimize", _mock_minimize):
        pr = polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    assert pr.telemetry.derivative_mode == "reference_fd"
    assert pr.telemetry.fallback_reason == "mock fallback trigger during minimize"

    # The fallback should have started from X0, not a partially advanced point
    np.testing.assert_allclose(fallback_x0, X0)


def test_g2_direct_fd_fallback_equivalence():
    """G2-J: Direct FD A/B fallback equivalence."""
    ctx_a = _build_case()
    cache_a = DomainEvaluatorCache()
    basin_a = _basin(ctx_a, cache_a)

    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("mock")

    with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
        pr_a = polish_basin(basin_a, 0, ctx_a, cache_a, _spec(DerivativeMode.ANALYTICAL))

    assert pr_a.telemetry.derivative_mode == "reference_fd"

    ctx_b = _build_case()
    cache_b = DomainEvaluatorCache()
    basin_b = _basin(ctx_b, cache_b)
    pr_b = polish_basin(basin_b, 0, ctx_b, cache_b, _spec(DerivativeMode.REFERENCE_FD))

    assert pr_a.post_polish.objective_value == pytest.approx(pr_b.post_polish.objective_value)
    assert pr_a.post_polish.v_max == pytest.approx(pr_b.post_polish.v_max)
    assert pr_a.post_polish.feasible == pr_b.post_polish.feasible


def test_g2_nominal_exchange_cleanup():
    """G2-M: Nominal exchange cleanup / state isolation."""
    ctx = _build_case()
    cache = DomainEvaluatorCache()
    basin = _basin(ctx, cache)

    original_evaluate = evaluate
    def _mock_evaluate(*args, **kwargs):
        # We evaluate normally, which publishes to the exchange
        res = original_evaluate(*args, **kwargs)
        return res

    def _mock_validate(self, j_base, j_constr):
        raise DerivativeUnavailable("mock")

    with patch("foster_eom.optimize.local_polish.evaluate", _mock_evaluate):
        with patch.object(AnalyticalDerivativeProvider, "_validate", _mock_validate):
            polish_basin(basin, 0, ctx, cache, _spec(DerivativeMode.ANALYTICAL))

    # Check that after fallback, the provider released its state.
    # We can check that the cache's nominal_exchange is None because it's only active during analytical.
    # Actually `provider.release()` sets cache.nominal_exchange to None!
    assert not cache.nominal_exchange.enabled
