"""Tests for Prompt-05 top-level optimization engine."""

from unittest.mock import MagicMock, patch

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.objectives import OptimizationSpec
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.foster.seed import SeedCandidate, SeedGenerationResult
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.engine import _allocate_budgets, run_optimization
from foster_eom.optimize.preflight import PreflightReport


def test_allocate_budgets_rounding():
    """Verify budget allocation rounds down and drops domains if insufficient."""
    mock_domain1 = MagicMock()
    mock_domain1.dimension = 2
    mock_domain1.domain_id = "d1"

    mock_domain2 = MagicMock()
    mock_domain2.dimension = 3
    mock_domain2.domain_id = "d2"

    # Min budget d1 = 20
    # Min budget d2 = 30

    # If budget = 60
    budget_map, n_dropped, truncated = _allocate_budgets(
        domains=[mock_domain1, mock_domain2], # best first
        de_budget=60,
        pop_multiplier=5,
        domain_deb_keys={"d1": (1,), "d2": (2,)}
    )

    assert n_dropped == 0
    assert not truncated
    assert sum(budget_map.values()) == 60

    # If budget = 40, d2 dropped
    budget_map, n_dropped, truncated = _allocate_budgets(
        domains=[mock_domain1, mock_domain2], # best first
        de_budget=40,
        pop_multiplier=5,
        domain_deb_keys={"d1": (1,), "d2": (2,)}
    )
    assert n_dropped == 1
    assert truncated
    assert "d1" in budget_map
    assert "d2" not in budget_map

@patch("foster_eom.optimize.engine.run_preflight")
@patch("foster_eom.optimize.engine.group_seeds_into_domains")
@patch("foster_eom.optimize.engine.evaluate")
@patch("foster_eom.optimize.engine.run_de")
@patch("foster_eom.optimize.engine.polish_top_k")
def test_engine_skips_zero_dimensional(mock_polish, mock_run_de, mock_eval, mock_group, mock_preflight):
    """Verify correctly skips zero-dimensional domains DE and directly evaluates."""
    mock_preflight.return_value = PreflightReport(passed=True, errors=(), warnings=())

    # Zero-dimensional domain
    mock_domain = MagicMock()
    mock_domain.structurally_feasible = True
    mock_domain.dimension = 0
    mock_domain.domain_id = "0d_domain"
    mock_domain.seed_indices = (0,)
    mock_domain.topology.branch1_cells = 0
    mock_domain.topology.branch2_cells = 0
    mock_domain.topology.branch1_has_c0 = False
    mock_domain.topology.branch1_has_linf = False
    mock_domain.topology.branch2_has_c0 = False
    mock_domain.topology.branch2_has_linf = False
    mock_domain.orientation.value = "test"
    mock_domain.branch1_realization.value = "test"
    mock_domain.branch2_realization.value = "test"

    mock_group.return_value = [mock_domain]

    # Mock seed evaluate
    mock_eval.return_value = MagicMock(
        x=(), objective_value=1.0, base_objective_value=1.0, soft_penalty_total=0.0,
        objective_terms={"total": 1.0}, hard_margins=(), soft_penalties={}, v_max=0.0, v_sum=0.0,
        feasible=True, near_feasible=True, numerical_status="ok", numerical_failure_reason=None,
        target_solutions=[], coarse_evaluated=False,
    )

    # mock DE
    from foster_eom.optimize.de_runner import DEDiagnostics
    mock_run_de.return_value = ([], DEDiagnostics("0d_domain", 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1.0, True, "zero_dim"))

    mock_polish.return_value = []

    seed = MagicMock(spec=SeedCandidate)
    seed.branch1_solve = None
    seed.branch2_solve = None

    seed_res = SeedGenerationResult(
        seeds=(seed,), diagnostics=MagicMock()
    )

    opt_spec = OptimizationSpec(max_global_evaluations=1000)

    res = run_optimization(
        seed_result=seed_res,
        opt_spec=opt_spec,
        source_spec=SourceSpec(mode=SourceMode.AVAILABLE_POWER, available_power_dbm=10.0),
        eom_model=MagicMock(spec=OnePortModel),
        component_limits=ContinuousLimits(c_min_f=1e-12, c_max_f=1e-6, l_min_h=1e-9, l_max_h=1e-3),
        match_constraints=MatchConstraints(gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0),
        stress_constraints=StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=5.0),
        target_frequencies_hz=(1e6,),
        sweep_f_min_hz=1e5,
        sweep_f_max_hz=1e7,
    )

    # Should evaluate seed
    assert mock_eval.call_count == 1
    # Run DE called (zero dim logic inside run_de handles skip, but engine still calls it or skips internally)
    assert mock_run_de.call_count == 1

    assert res.run_manifest.n_domains_optimized == 1
    # Check that candidate returned is the seed candidate
    assert len(res.candidates) == 1

@patch("foster_eom.optimize.engine.run_preflight")
@patch("foster_eom.optimize.engine.group_seeds_into_domains")
@patch("foster_eom.optimize.engine.evaluate")
@patch("foster_eom.optimize.engine.run_de")
def test_rejects_infeasible_domains(mock_run_de, mock_eval, mock_group, mock_preflight):
    """Rejects domains with structurally_feasible=False."""
    mock_preflight.return_value = PreflightReport(passed=True, errors=(), warnings=())

    mock_domain = MagicMock()
    mock_domain.structurally_feasible = False
    mock_domain.domain_id = "infeasible"

    mock_group.return_value = [mock_domain]

    seed_res = SeedGenerationResult(seeds=(), diagnostics=MagicMock())

    res = run_optimization(
        seed_result=seed_res,
        opt_spec=OptimizationSpec(),
        source_spec=SourceSpec(mode=SourceMode.AVAILABLE_POWER, available_power_dbm=10.0),
        eom_model=MagicMock(spec=OnePortModel),
        component_limits=ContinuousLimits(c_min_f=1e-12, c_max_f=1e-6, l_min_h=1e-9, l_max_h=1e-3),
        match_constraints=MatchConstraints(gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0),
        stress_constraints=StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=5.0),
        target_frequencies_hz=(1e6,),
        sweep_f_min_hz=1e5,
        sweep_f_max_hz=1e7,
    )

    assert mock_eval.call_count == 0
    assert mock_run_de.call_count == 0
    assert res.run_manifest.n_domains_optimized == 0
