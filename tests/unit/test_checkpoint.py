"""Tests for Checkpoint and Warm-Restart in Prompt-05."""

from unittest.mock import MagicMock, patch

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.objectives import OptimizationSpec
from foster_eom.domain.results import CandidateResult
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.foster.seed import SeedCandidate, SeedGenerationResult
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.engine import run_optimization
from foster_eom.optimize.preflight import PreflightReport


@patch("foster_eom.persistence.yaml_io.save_results")
@patch("foster_eom.optimize.engine.run_preflight")
@patch("foster_eom.optimize.engine.group_seeds_into_domains")
@patch("foster_eom.optimize.engine.evaluate")
@patch("scipy.optimize.differential_evolution")
def test_checkpointing(mock_de, mock_eval, mock_group, mock_preflight, mock_save, tmp_path):
    """Verify that a checkpoint YAML is written every N evaluations."""
    mock_preflight.return_value = PreflightReport(passed=True, errors=(), warnings=())

    mock_domain = MagicMock()
    mock_domain.structurally_feasible = True
    mock_domain.dimension = 2
    mock_domain.domain_id = "test_domain"
    mock_domain.seed_indices = (0,)
    mock_domain.orientation.value = "test"
    mock_domain.branch1_realization.value = "test"
    mock_domain.branch2_realization.value = "test"
    mock_group.return_value = [mock_domain]

    seed = MagicMock(spec=SeedCandidate)
    seed.branch1_solve = None
    seed.branch2_solve = None

    seed_res = SeedGenerationResult(seeds=(seed,), diagnostics=MagicMock())

    opt_spec = OptimizationSpec(max_global_evaluations=1000, checkpoint_every_evaluations=100)

    mock_eval.return_value = MagicMock(
        x=(0.1, 0.2),
        objective_value=1.0,
        base_objective_value=1.0,
        soft_penalty_total=0.0,
        objective_terms={"total": 1.0},
        hard_margins=(),
        soft_penalties={},
        v_max=0.0,
        v_sum=0.0,
        feasible=True,
        near_feasible=True,
        numerical_status="ok",
        numerical_failure_reason=None,
        target_solutions=[],
        coarse_evaluated=False,
    )

    mock_cache = MagicMock()
    mock_cache._cache = {"dummy_key": mock_eval.return_value}

    def side_effect_de(*args, **kwargs):
        # simulate some evaluations by advancing cache
        # the callback checks cache.n_unique_evaluations
        callback = kwargs.get("callback")
        if callback:
            mock_cache.n_unique_evaluations += 150
            callback(MagicMock(x=[0.1, 0.2]))
        return MagicMock(x=np.array([0.1, 0.2]), message="ok", success=True)

    mock_de.side_effect = side_effect_de

    checkpoint_file = tmp_path / "checkpoint.yaml"

    # Needs to bypass actual evaluation if it accesses mock cache, but let's see
    with patch("foster_eom.optimize.engine.DomainEvaluatorCache") as mock_cache_cls:
        # Set evaluation count low before DE starts
        mock_cache.n_unique_evaluations = 0
        mock_cache_cls.return_value = mock_cache

        run_optimization(
            seed_result=seed_res,
            opt_spec=opt_spec,
            source_spec=SourceSpec(mode=SourceMode.AVAILABLE_POWER, available_power_dbm=10.0),
            eom_model=MagicMock(spec=OnePortModel),
            component_limits=ContinuousLimits(
                c_min_f=1e-12, c_max_f=1e-6, l_min_h=1e-9, l_max_h=1e-3
            ),
            match_constraints=MatchConstraints(
                gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0
            ),
            stress_constraints=StressConstraints(
                source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=5.0
            ),
            target_frequencies_hz=(1e6,),
            sweep_f_min_hz=1e5,
            sweep_f_max_hz=1e7,
            checkpoint_path=checkpoint_file,
        )

    # verify save_results was called for checkpoint
    assert mock_save.call_count >= 1
    call_args = mock_save.call_args[0]
    assert call_args[1] == checkpoint_file


def test_warm_restarts():
    """Verify warm start candidates can be passed."""
    from foster_eom.optimize.de_runner import _build_initial_population

    seed_vecs = [np.array([0.1, 0.2])]
    cand = CandidateResult(candidate_id="w1", feasible=True)

    # In _build_initial_population, it checks warm_start_candidates.
    # Currently it's a pass/stub but should accept the parameter without crashing.
    pop = _build_initial_population(
        analytic_x_vectors=seed_vecs,
        n_pop=10,
        n_dim=2,
        random_seed=42,
        domain_id="d1",
        warm_start_candidates=[cand],
    )
    assert pop.shape == (10, 2)
