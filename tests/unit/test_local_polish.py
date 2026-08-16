"""Tests for Prompt-05 local polish."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from foster_eom.domain.objectives import LocalMethod, OptimizationSpec
from foster_eom.optimize.dedup import Basin
from foster_eom.optimize.evaluator import DomainEvaluatorCache, EvaluationResult
from foster_eom.optimize.local_polish import (
    _resolve_method,
    polish_basin,
)


def test_resolve_method():
    """Verify TRUST_CONSTR, IPOPT fallback, and SLSQP rejection."""
    spec = OptimizationSpec(
        local_method=LocalMethod.TRUST_CONSTR,
        local_fallback_method=LocalMethod.TRUST_CONSTR,
    )
    assert _resolve_method(spec) == "trust-constr"

    spec_slsqp = OptimizationSpec(local_method=LocalMethod.SLSQP)
    with pytest.raises(RuntimeError):
        _resolve_method(spec_slsqp)

    spec_ipopt = OptimizationSpec(local_method=LocalMethod.IPOPT)
    # Since cyipopt is not installed in the typical environment, it falls back
    # But _resolve_method always returns trust-constr for IPOPT fallback according to current implementation.
    # Actually wait, _resolve_method:
    # if primary == IPOPT... except ImportError -> method = fallback
    # if method == TRUST_CONSTR return "trust-constr"
    # if method == IPOPT return "trust-constr"  (last resort)
    # return "trust-constr" (default)
    # So it always returns trust-constr
    assert _resolve_method(spec_ipopt) == "trust-constr"

@patch("scipy.optimize.minimize")
def test_polish_deb_worse_discarded(mock_minimize):
    """Verify Deb-worse polish result is discarded in favor of pre-polish."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 2
    mock_context.domain.domain_id = "test"
    cache = DomainEvaluatorCache()
    spec = OptimizationSpec()

    pre = EvaluationResult(
        x=(0.1, 0.2), objective_value=10.0, base_objective_value=10.0, soft_penalty_total=0.0,
        objective_terms={"total": 10.0}, hard_margins=(1.0,), soft_penalties={}, v_max=0.0, v_sum=0.0,
        feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
        failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False,
    )

    basin = Basin(representative=pre, members=[pre])

    # Mock minimize to return a worse result
    mock_minimize.return_value = MagicMock(x=np.array([0.5, 0.5]), success=True, nit=10, message="ok")

    # We also need to mock evaluate to return the worse result for the post_x
    with patch("foster_eom.optimize.local_polish.evaluate") as mock_eval:
        post = EvaluationResult(
            x=(0.5, 0.5), objective_value=20.0, base_objective_value=20.0, soft_penalty_total=0.0,
            objective_terms={"total": 20.0}, hard_margins=(1.0,), soft_penalties={}, v_max=0.0, v_sum=0.0,
            feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
            failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False,
        )
        mock_eval.return_value = post

        pr = polish_basin(basin, 0, mock_context, cache, spec)

        assert pr.success is True
        assert pr.post_polish is post
        assert pr.retained is pre  # pre is better (10.0 < 20.0), so retained
        assert mock_minimize.call_count == 1

@patch("scipy.optimize.minimize")
def test_polish_deb_better_retained(mock_minimize):
    """Verify Deb-better polish result is retained."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 2
    mock_context.domain.domain_id = "test"
    cache = DomainEvaluatorCache()
    spec = OptimizationSpec()

    pre = EvaluationResult(
        x=(0.1, 0.2), objective_value=20.0, base_objective_value=20.0, soft_penalty_total=0.0,
        objective_terms={"total": 20.0}, hard_margins=(1.0,), soft_penalties={}, v_max=0.0, v_sum=0.0,
        feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
        failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False,
    )

    basin = Basin(representative=pre, members=[pre])

    mock_minimize.return_value = MagicMock(x=np.array([0.5, 0.5]), success=True, nit=10, message="ok")

    with patch("foster_eom.optimize.local_polish.evaluate") as mock_eval:
        post = EvaluationResult(
            x=(0.5, 0.5), objective_value=10.0, base_objective_value=10.0, soft_penalty_total=0.0,
            objective_terms={"total": 10.0}, hard_margins=(1.0,), soft_penalties={}, v_max=0.0, v_sum=0.0,
            feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
            failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False,
        )
        mock_eval.return_value = post

        pr = polish_basin(basin, 0, mock_context, cache, spec)

        assert pr.success is True
        assert pr.retained is post

def test_zero_dimensional_bypasses_polish():
    """Verify zero-dimensional domains bypass local polish."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 0
    mock_context.domain.domain_id = "test"
    cache = DomainEvaluatorCache()
    spec = OptimizationSpec()

    pre = EvaluationResult(
        x=(), objective_value=10.0, base_objective_value=10.0, soft_penalty_total=0.0,
        objective_terms={"total": 10.0}, hard_margins=(1.0,), soft_penalties={}, v_max=0.0, v_sum=0.0,
        feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
        failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False,
    )

    basin = Basin(representative=pre, members=[pre])

    with patch("scipy.optimize.minimize") as mock_minimize:
        pr = polish_basin(basin, 0, mock_context, cache, spec)
        assert mock_minimize.call_count == 0
        assert pr.success is True
        assert pr.retained is pre
        assert pr.termination == "zero_dimensional"

@patch("scipy.optimize.minimize", side_effect=Exception("numerical failure"))
def test_numerical_solver_failure_captured(mock_minimize):
    """Verify numerical solver failure is captured structurally."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 2
    mock_context.domain.domain_id = "test"
    cache = DomainEvaluatorCache()
    spec = OptimizationSpec()

    pre = EvaluationResult(
        x=(0.1, 0.2), objective_value=10.0, base_objective_value=10.0, soft_penalty_total=0.0,
        objective_terms={"total": 10.0}, hard_margins=(1.0,), soft_penalties={}, v_max=0.0, v_sum=0.0,
        feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
        failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False,
    )

    basin = Basin(representative=pre, members=[pre])

    pr = polish_basin(basin, 0, mock_context, cache, spec)
    assert not pr.success
    assert pr.retained is pre
    assert "Exception" in pr.termination
    assert pr.reason == "numerical failure"
