"""P12-GUI2 Schema/Provenance Regression Tests.

Tests that OptimizeVM.from_result() correctly maps every field from
the *current* CandidateResult schema, and that the full
OptimizeCtrl → OptimizeVM → SynthesizePage pipeline does not crash.

These tests use real schema instances (no mocking) to catch stale
field renames that automated unit tests missed.
"""

from __future__ import annotations

from foster_eom.domain.results import (
    CandidateResult,
    CoarseGridSummary,
    TargetSolutionSummary,
)
from foster_eom.gui.view_models.optimize_vm import OptimizeVM
from foster_eom.optimize.engine import OptimizationResult, RunManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_candidate(**overrides) -> CandidateResult:
    """Build a real CandidateResult from the current schema with no mocking."""
    defaults = dict(
        candidate_id="test_001",
        topology_id="topo_a",
        orientation="shunt",
        domain_id="dom1",
        branch1_realization="foster",
        branch2_realization="foster",
        branch1_cells=1,
        branch2_cells=0,
        branch1_has_c0=False,
        branch1_has_linf=False,
        branch2_has_c0=False,
        branch2_has_linf=False,
        continuous_variables={"k1": 0.5},
        resolved_values={"L1": 1e-6, "C1": 10e-12},
        catalog_parts={},
        pole_locations_hz=[10e6],
        objective_terms={"total": 0.123, "base": 0.100, "soft_penalty": 0.023, "j_gamma": 0.05},
        constraint_margins={"hard_0": 0.1, "hard_1": -0.02},
        warnings=[],
        solver_diagnostics={},
        feasible=True,
        near_feasible=True,
        v_max=0.15,
        v_sum=0.30,
        base_objective_value=0.100,
        soft_penalty_total=0.023,
        numerical_status="ok",
        k_residues_branch1=[0.5],
        k_residues_branch2=[],
        k0_branch1=None,
        k_inf_branch1=None,
        k0_branch2=None,
        k_inf_branch2=None,
        pole_frequencies_branch1_hz=[10e6],
        pole_frequencies_branch2_hz=[],
        target_solution_summaries=[
            TargetSolutionSummary(
                frequency_hz=10e6,
                z_in_real=50.0,
                z_in_imag=0.5,
                gamma_mag=0.01,
                s11_db=-40.0,
                v_eom_mag=5.0,
                i_source_rms=0.02,
                power_balance_ok=True,
            )
        ],
        coarse_grid_summary=CoarseGridSummary(
            coarse_evaluated=True,
            off_target_n_points=100,
            off_target_v_eom_peak_v=15.0,
        ),
        seed_source="foster_schmidt_04b",
        de_domain_id="dom1",
        de_evaluations_used=500,
        de_generation_reached=25,
        pre_polish_objective=0.15,
        local_polish_method="trust-constr",
        local_polish_outcome="polished_retained",
        local_polish_success=True,
        local_polish_iterations=42,
        local_polish_evaluations=120,
        solver_termination="converged",
    )
    defaults.update(overrides)
    return CandidateResult(**defaults)


def _real_manifest() -> RunManifest:
    return RunManifest(
        foster_eom_version="0.12.0-test",
        numpy_version="2.4.6",
        scipy_version="1.18.0",
        random_seed=42,
        requested_global_budget=500,
        seed_evaluation_budget_used=10,
        de_budget_available=490,
        allocated_budget_per_domain={"dom1": 490},
        unique_x_evaluations_per_domain={"dom1": 480},
        total_unique_x_evaluations=480,
        budget_exhausted=True,
        n_domains_available=1,
        n_domains_selected_before_budget=1,
        n_domains_optimized=1,
        n_domains_dropped_for_budget=0,
        domain_search_truncated=False,
    )


def _real_optimization_result(**overrides) -> OptimizationResult:
    c1 = _real_candidate(candidate_id="c1", feasible=True, base_objective_value=0.05,
                         soft_penalty_total=0.01)
    c2 = _real_candidate(candidate_id="c2", feasible=False, near_feasible=True,
                         base_objective_value=0.10, soft_penalty_total=0.02)
    c3 = _real_candidate(candidate_id="c3", feasible=False, near_feasible=False,
                         base_objective_value=0.50, soft_penalty_total=0.05)

    # OptimizeVM.from_result only accesses candidates, best_feasible,
    # near_feasible_best, and run_manifest. Mock the diagnostic fields
    # since they have deep frozen schemas irrelevant to the view model.
    from unittest.mock import MagicMock

    defaults = dict(
        candidates=(c1, c2, c3),
        best_feasible=c1,
        near_feasible_best=c2,
        preflight=MagicMock(),
        seed_diagnostics=MagicMock(),
        de_diagnostics=(),
        run_manifest=_real_manifest(),
    )
    defaults.update(overrides)
    return OptimizationResult(**defaults)


# ===========================================================================
# 1. OptimizeVM.from_result() with real CandidateResult (no mocking)
# ===========================================================================


class TestOptimizeVMFromRealResult:
    """Regression: from_result() must not crash on the current schema."""

    def test_from_result_does_not_crash(self):
        """The primary regression: construct real result → from_result()."""
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        assert vm is not None

    def test_status_label_feasible(self):
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        assert vm.status_label == "FEASIBLE"

    def test_status_label_near_feasible(self):
        result = _real_optimization_result(best_feasible=None)
        vm = OptimizeVM.from_result(result)
        assert vm.status_label == "NEAR-FEASIBLE"

    def test_status_label_infeasible(self):
        result = _real_optimization_result(best_feasible=None, near_feasible_best=None)
        vm = OptimizeVM.from_result(result)
        assert vm.status_label == "INFEASIBLE"

    def test_candidate_count(self):
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        assert len(vm.candidates) == 3

    def test_candidate_ranks(self):
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        assert [c.rank for c in vm.candidates] == [1, 2, 3]

    def test_objective_is_j_total(self):
        """Objective column = base_objective_value + soft_penalty_total."""
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        c1 = vm.candidates[0]
        assert abs(c1.objective - (0.05 + 0.01)) < 1e-10

    def test_feasibility_mapped_correctly(self):
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        assert vm.candidates[0].feasible is True
        assert vm.candidates[1].feasible is False
        assert vm.candidates[2].feasible is False

    def test_near_feasibility_mapped_correctly(self):
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        assert vm.candidates[0].near_feasible is True
        assert vm.candidates[1].near_feasible is True
        assert vm.candidates[2].near_feasible is False

    def test_numerical_status_is_string(self):
        """numerical_status is already str, not an enum — no .value needed."""
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        for c in vm.candidates:
            assert isinstance(c.numerical_status, str)
            assert c.numerical_status == "ok"


# ===========================================================================
# 2. Provenance field mapping: RunManifest fields used in SynthesizePage
# ===========================================================================


class TestRunManifestProvenance:
    """SynthesizePage._on_finished accesses run_manifest.requested_global_budget
    and run_manifest.random_seed for the status label."""

    def test_manifest_fields_accessible(self):
        m = _real_manifest()
        assert m.requested_global_budget == 500
        assert m.random_seed == 42

    def test_manifest_formatted_in_status_line(self):
        """Simulates the status-line formatting from _on_finished."""
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        spec = result.run_manifest
        status = (
            f"Done — {len(vm.candidates)} candidates | "
            f"Preset: FAST | "
            f"Global evals: {spec.requested_global_budget:,} | "
            f"Seed: {spec.random_seed}"
        )
        assert "3 candidates" in status
        assert "500" in status
        assert "42" in status


# ===========================================================================
# 3. Integration: OptimizeCtrl result → OptimizeVM → SynthesizePage rendering
# ===========================================================================


class TestSynthesizePageResultRendering:
    """Regression: a successful optimization must not crash on display."""

    def test_candidate_row_fields_are_correct_types(self):
        """All CandidateRow fields must have correct types for QTableWidgetItem."""
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        for c in vm.candidates:
            assert isinstance(c.rank, int)
            assert isinstance(c.objective, float)
            assert isinstance(c.feasible, bool)
            assert isinstance(c.near_feasible, bool)
            assert isinstance(c.numerical_status, str)

    def test_table_item_formatting(self):
        """Simulates the exact formatting done in SynthesizePage._on_finished."""
        result = _real_optimization_result()
        vm = OptimizeVM.from_result(result)
        for c in vm.candidates:
            # These are the exact expressions from synthesize_page.py L298-302
            str(c.rank)         # column 0
            f"{c.objective:.6f}"  # column 1
            "✓" if c.feasible else "✗"  # column 2
            "✓" if c.near_feasible else "✗"  # column 3
            c.numerical_status  # column 4  (must be str, not enum)

    def test_detail_text_rendering(self):
        """Simulates _on_selection detail rendering against real CandidateResult.

        Updated for P12-GUI2 closeout: uses CandidateDetailVM + constraint summary.
        """
        c = _real_candidate()
        from foster_eom.gui.view_models.optimize_vm import (
            CandidateDetailVM,
            format_polish_provenance,
        )
        vm = CandidateDetailVM.from_candidate(1, c)
        polish_lines = format_polish_provenance(vm.local_polish_method, vm.local_polish_outcome)
        # Key assertions: provenance is non-blank and contains method
        assert any("trust-constr" in line for line in polish_lines)
        assert any("polished candidate retained" in line for line in polish_lines)
        # Constraint summary uses descriptor names (not hard_N)
        assert vm.total_hard == len(c.constraint_margins)
        assert vm.violated_count == len([v for v in c.constraint_margins.values() if v < 0])


# ===========================================================================
# 4. CandidateResult schema completeness: no stale accesses elsewhere
# ===========================================================================


class TestCandidateResultSchemaCompleteness:
    """Guard against field renames: all fields used by GUI must exist."""

    def test_all_gui_accessed_fields_exist(self):
        """Every field accessed by optimize_vm.py and synthesize_page.py."""
        c = _real_candidate()
        # optimize_vm.py
        _ = c.base_objective_value
        _ = c.soft_penalty_total
        _ = c.feasible
        _ = c.near_feasible
        _ = c.numerical_status
        # synthesize_page.py _on_selection
        _ = c.topology_id
        _ = c.v_max
        _ = c.local_polish_method
        _ = c.local_polish_outcome
        _ = c.seed_source
        _ = c.objective_terms
        _ = c.constraint_margins

    def test_numerical_status_is_str_not_enum(self):
        c = _real_candidate()
        assert isinstance(c.numerical_status, str)
        assert not hasattr(c.numerical_status, "value") or isinstance(c.numerical_status, str)

    def test_objective_value_not_on_candidate_result(self):
        """objective_value is on EvaluationResult, NOT CandidateResult."""
        c = _real_candidate()
        assert not hasattr(c, "objective_value"), (
            "CandidateResult should NOT have objective_value — "
            "use base_objective_value + soft_penalty_total instead"
        )
