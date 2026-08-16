"""Tests for Prompt-05 Results Persistence."""

from pathlib import Path

from foster_eom.domain.results import CandidateResult, CoarseGridSummary, TargetSolutionSummary
from foster_eom.foster.seed import SeedGenerationResult
from foster_eom.optimize.de_runner import DEDiagnostics
from foster_eom.optimize.engine import OptimizationResult, RunManifest
from foster_eom.optimize.preflight import PreflightReport
from foster_eom.persistence.yaml_io import load_results, save_results


def test_save_load_optimization_result(tmp_path: Path):
    """Verify CandidateResult schema and RunManifest are losslessly serializable."""
    cand = CandidateResult(
        candidate_id="c1",
        domain_id="d1",
        v_max=1.0,
        v_sum=2.0,
        objective_terms={"total": 5.0},
        feasible=True,
        target_solution_summaries=[
            TargetSolutionSummary(frequency_hz=1e6, z_in_real=50.0, s11_db=-20.0)
        ],
        coarse_grid_summary=CoarseGridSummary(coarse_evaluated=True, off_target_n_points=100)
    )

    preflight = PreflightReport(passed=True, errors=(), warnings=())
    from foster_eom.foster.seed import SeedGenerationDiagnostics
    diag = SeedGenerationDiagnostics(
        n_orientation_attempts=1, n_sign_patterns=1, n_topologies=1,
        n_pole_layouts_branch1=1, n_pole_layouts_branch2=1, n_pole_layout_pairs=1,
        n_solver_attempts=1, n_mna_attempts=1, rejection_counts={},
        representative_failures=(), max_failure_records_per_code=1,
        sign_search_by_orientation={}, sign_search_exhaustive=True,
        sign_search_truncated=False, sign_beam_width=1, sign_max_patterns=1
    )
    seed_diag = SeedGenerationResult(seeds=(), diagnostics=diag)
    de_diag = DEDiagnostics(
        domain_id="d1", n_pop=10, n_gen_requested=50, n_gen_completed=50, budget_allocated=500,
        unique_x_evaluations=500, cache_hits=0, target_frequency_point_solves=500,
        coarse_frequency_point_solves=500, total_frequency_point_solves=1000,
        numerical_failures=0, best_objective=5.0, best_feasible=True, de_termination="ok"
    )

    manifest = RunManifest(
        foster_eom_version="0.1.0",
        numpy_version="1.21.0",
        scipy_version="1.7.0",
        random_seed=42,
        requested_global_budget=1000,
        seed_evaluation_budget_used=10,
        de_budget_available=990,
        allocated_budget_per_domain={"d1": 500},
        unique_x_evaluations_per_domain={"d1": 500},
        total_unique_x_evaluations=510,
        budget_exhausted=False,
        n_domains_available=1,
        n_domains_selected_before_budget=1,
        n_domains_optimized=1,
        n_domains_dropped_for_budget=0,
        domain_search_truncated=False
    )

    result = OptimizationResult(
        candidates=(cand,),
        best_feasible=cand,
        near_feasible_best=cand,
        preflight=preflight,
        seed_diagnostics=seed_diag,
        de_diagnostics=(de_diag,),
        run_manifest=manifest
    )

    file_path = tmp_path / "results.yaml"

    # Save
    save_results(result, file_path)
    assert file_path.exists()

    # Load
    loaded = load_results(file_path)

    # Verify CandidateResult
    assert len(loaded.candidates) == 1
    c_loaded = loaded.candidates[0]
    assert c_loaded.candidate_id == "c1"
    assert c_loaded.feasible is True
    assert c_loaded.v_max == 1.0
    assert len(c_loaded.target_solution_summaries) == 1
    assert c_loaded.target_solution_summaries[0].z_in_real == 50.0

    # Verify Manifest
    m_loaded = loaded.run_manifest
    assert m_loaded.foster_eom_version == "0.1.0"
    assert m_loaded.random_seed == 42
    assert m_loaded.allocated_budget_per_domain == {"d1": 500}

    # Verify DE diag
    assert len(loaded.de_diagnostics) == 1
    d_loaded = loaded.de_diagnostics[0]
    assert d_loaded.domain_id == "d1"
    assert d_loaded.n_pop == 10
