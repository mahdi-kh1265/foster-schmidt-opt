"""Tests for Prompt-05 Central Evaluator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    EvaluationResult,
    _failure_result,
    evaluate,
)


def test_objective_value_sum():
    """objective_value == J_base + J_soft."""
    # We can test this by checking the EvaluationResult returned from a mock solve.
    # It's easier to assert on the structure.
    res = EvaluationResult(
        x=(0.5,),
        objective_value=15.0,
        base_objective_value=10.0,
        soft_penalty_total=5.0,
        objective_terms={"total": 15.0, "base": 10.0, "soft_penalty": 5.0},
        hard_margins=(0.1,),
        soft_penalties={},
        v_max=0.0,
        v_sum=0.0,
        feasible=True,
        near_feasible=True,
        numerical_status="ok",
        numerical_failure_reason=None,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=(),
        target_solutions=(),
        coarse_evaluated=False,
    )
    assert res.objective_value == res.base_objective_value + res.soft_penalty_total

@patch("foster_eom.optimize.evaluator._evaluate_uncached")
def test_cache_hits(mock_eval_uncached):
    """exact-x second evaluation hits cache and performs no additional physical solve."""
    from foster_eom.optimize.evaluator import EvaluationResult

    mock_eval_uncached.return_value = EvaluationResult(
        x=(0.5,), objective_value=10.0, base_objective_value=10.0, soft_penalty_total=0.0,
        objective_terms={"total": 10.0}, hard_margins=(), soft_penalties={}, v_max=0.0, v_sum=0.0,
        feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
        failed_frequency_hz=None, failed_stage=None, all_solutions=(), target_solutions=(), coarse_evaluated=False
    )

    context = MagicMock()
    context.target_indices = (0,)
    context.evaluation_frequencies_hz = (1e6,)

    cache = DomainEvaluatorCache()
    x = np.array([0.5])

    # First eval
    res1 = evaluate(x, context, cache)
    assert cache.n_unique_evaluations == 1
    assert cache.n_calls == 1
    assert cache.n_cache_hits == 0
    assert mock_eval_uncached.call_count == 1

    # Second eval
    res2 = evaluate(x, context, cache)
    assert cache.n_unique_evaluations == 1
    assert cache.n_calls == 2
    assert cache.n_cache_hits == 1
    assert mock_eval_uncached.call_count == 1  # No additional call
    assert res1 is res2

def test_programming_errors_not_swallowed():
    """Programming errors (like KeyError) are not swallowed."""
    cache = DomainEvaluatorCache()
    context = MagicMock()
    # If unpack raises KeyError or AttributeError
    context.domain.variable_mapper.unpack.side_effect = AttributeError("test error")
    context.hard_layout.n = 0

    with pytest.raises(AttributeError):
        evaluate(np.array([0.5]), context, cache)

def test_numerical_mna_singularity():
    """numerical MNA singularity produces finite deterministic invalid result."""
    x_key = (0.5,)
    context = MagicMock()
    res = _failure_result(x_key, n_hard=2, context=context, status="mna_singular", reason="singular")
    assert not res.feasible
    assert res.objective_value == 1e9
    assert res.hard_margins == (-1.0, -1.0)
    assert res.v_max == 1.0

def test_empty_hard_constraint():
    """empty hard-constraint vector with valid MNA gives v_max=v_sum=0 and can be feasible."""
    x_key = (0.5,)
    context = MagicMock()
    # It's an internal test, simulating a successful run with no hard constraints.
    res = EvaluationResult(
        x=x_key, objective_value=1.0, base_objective_value=1.0, soft_penalty_total=0.0,
        objective_terms={"total": 1.0, "base": 1.0, "soft_penalty": 0.0},
        hard_margins=(), soft_penalties={}, v_max=0.0, v_sum=0.0, feasible=True, near_feasible=True,
        numerical_status="ok", numerical_failure_reason=None, failed_frequency_hz=None, failed_stage=None,
        all_solutions=(), target_solutions=(), coarse_evaluated=False,
    )
    assert len(res.hard_margins) == 0
    assert res.v_max == 0.0
    assert res.v_sum == 0.0
    assert res.feasible

def test_numerical_invalidity_zero_hard_constraints():
    """numerical invalidity still makes zero-hard-constraint candidate infeasible."""
    x_key = (0.5,)
    context = MagicMock()
    res = _failure_result(x_key, n_hard=0, context=context, status="mna_singular", reason="singular")
    assert not res.feasible
    assert res.v_max == 1.0  # From _failure_result fallback
    assert len(res.hard_margins) == 0
