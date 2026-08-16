"""Tests for Prompt-05 DE runner."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from foster_eom.optimize.de_runner import (
    _build_initial_population,
    resolve_workers,
    run_de,
)
from foster_eom.optimize.evaluator import DomainEvaluatorCache, EvaluationResult


def test_resolve_workers():
    assert resolve_workers(1) == 1
    assert resolve_workers(4) == 4
    assert resolve_workers("auto") >= 1
    with pytest.raises(ValueError):
        resolve_workers(0)
    with pytest.raises(ValueError):
        resolve_workers("foo")


def test_build_initial_population():
    """Verify analytic seeds, perturbations, and Sobol fill."""
    seed_vecs = [
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
    ]

    pop = _build_initial_population(
        analytic_x_vectors=seed_vecs,
        n_pop=10,
        n_dim=2,
        random_seed=42,
        domain_id="test_domain",
    )

    assert pop.shape == (10, 2)
    # The first row MUST be exactly the first seed
    assert np.allclose(pop[0], [0.1, 0.2])
    # The second row MUST be exactly the second seed
    assert np.allclose(pop[1], [0.3, 0.4])

    # Perturbations must remain within [0, 1]
    assert np.all(pop >= 0.0)
    assert np.all(pop <= 1.0)

    # Determinism
    pop2 = _build_initial_population(
        analytic_x_vectors=seed_vecs,
        n_pop=10,
        n_dim=2,
        random_seed=42,
        domain_id="test_domain",
    )
    assert np.allclose(pop, pop2)


@patch("scipy.optimize.differential_evolution")
def test_de_runner_calls_scipy_correctly(mock_de):
    """Verify polish=False, NonlinearConstraint, maxiter calculation."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 2
    mock_context.domain.domain_id = "test"
    cache = DomainEvaluatorCache()

    seed_res = EvaluationResult(
        x=(0.1, 0.2),
        objective_value=10.0,
        base_objective_value=10.0,
        soft_penalty_total=0.0,
        objective_terms={"total": 10.0, "base": 10.0, "soft_penalty": 0.0},
        hard_margins=(1.0,),
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

    budget = 100
    pop_mult = 5

    mock_de.return_value = MagicMock(message="ok")

    _results, _diag = run_de(
        context=mock_context,
        cache=cache,
        analytic_seed_results=[seed_res],
        budget=budget,
        population_size_multiplier=pop_mult,
        random_seed=42,
        de_strategy="best1bin",
        workers=1,
    )

    assert mock_de.call_count == 1
    kwargs = mock_de.call_args[1]

    # verify polish=False
    assert kwargs["polish"] is False

    # Verify NonlinearConstraint
    from scipy.optimize import NonlinearConstraint

    assert isinstance(kwargs["constraints"], NonlinearConstraint)

    # Verify maxiter
    max(1, 5 * 2)  # 10
    expected_gen = max(0, math.floor(budget / 10) - 1)
    assert kwargs["maxiter"] == expected_gen


def test_zero_dimensional_bypass():
    """Verify zero-dimensional domains bypass DE entirely."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 0
    mock_context.domain.domain_id = "test_0d"
    cache = DomainEvaluatorCache()

    seed_res = EvaluationResult(
        x=(),
        objective_value=10.0,
        base_objective_value=10.0,
        soft_penalty_total=0.0,
        objective_terms={"total": 10.0, "base": 10.0, "soft_penalty": 0.0},
        hard_margins=(1.0,),
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

    with patch("scipy.optimize.differential_evolution") as mock_de:
        results, diag = run_de(
            context=mock_context,
            cache=cache,
            analytic_seed_results=[seed_res],
            budget=1000,
            population_size_multiplier=15,
            random_seed=42,
            de_strategy="best1bin",
            workers=1,
        )
        assert mock_de.call_count == 0
        assert diag.de_termination == "zero_dimensional_fixed_evaluation"
        assert len(results) == 1


def test_de_failure_retains_seed():
    """Verify DE crash/failure retains the analytic seed."""
    mock_context = MagicMock()
    mock_context.domain.dimension = 2
    mock_context.domain.domain_id = "test"
    cache = DomainEvaluatorCache()

    seed_res = EvaluationResult(
        x=(0.1, 0.2),
        objective_value=10.0,
        base_objective_value=10.0,
        soft_penalty_total=0.0,
        objective_terms={"total": 10.0, "base": 10.0, "soft_penalty": 0.0},
        hard_margins=(1.0,),
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

    with patch("scipy.optimize.differential_evolution", side_effect=Exception("DE crash")):
        results, diag = run_de(
            context=mock_context,
            cache=cache,
            analytic_seed_results=[seed_res],
            budget=100,
            population_size_multiplier=15,
            random_seed=42,
            de_strategy="best1bin",
            workers=1,
        )
        assert "exception" in diag.de_termination
        assert len(results) == 1
        assert results[0].x == (0.1, 0.2)
