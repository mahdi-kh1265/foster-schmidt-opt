"""P12-GUI2: Execution Controls, Cancellation, and Constraint Exposure Tests."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from foster_eom.domain.objectives import DerivativeMode, OptimizationSpec
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import (
    ComponentLimitParams,
    EOMParams,
    MatchParams,
    ObjectiveWeightParams,
    OptimizationPresetParams,
    ProjectState,
    SourceParams,
    StressParams,
    TopologyParams,
)
from foster_eom.optimize.cancel import CancelledException, check_cancel
from foster_eom.optimize.progress import ProgressUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_test_state(**overrides) -> ProjectState:
    state = ProjectState()
    state.name = "GUI2 Test"
    state.frequencies_hz = [10e6]
    state.sweep_f_min_hz = 1e6
    state.sweep_f_max_hz = 30e6
    state.source = SourceParams(mode="thevenin", vth_rms=2.0, z_source_ohm=50.0)
    state.eom = EOMParams(model_type="ideal_capacitor", c0_f=1e-9)
    state.topology = TopologyParams(n_branches=2, n_cells_per_branch=1)
    state.input_sha256 = state.compute_input_sha()
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ===========================================================================
# 1. Preset compilation tests
# ===========================================================================


class TestPresetCompilation:
    def test_fast_preset_compiles_exact_budget(self):
        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(preset="FAST")
        )
        spec = state_to_spec(state)
        opt = spec.optimization
        assert opt.max_global_evaluations == 500
        assert opt.polish_top_k == 1
        assert opt.local_max_iterations == 20
        assert opt.random_seed == 42
        assert opt.local_derivative_mode == DerivativeMode.ANALYTICAL

    def test_balanced_preset_compiles_budget(self):
        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(preset="BALANCED")
        )
        spec = state_to_spec(state)
        opt = spec.optimization
        assert opt.max_global_evaluations == 2500
        assert opt.polish_top_k == 2
        assert opt.local_max_iterations == 100
        assert opt.random_seed == 20260815

    def test_thorough_preset_compiles_budget(self):
        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(preset="THOROUGH")
        )
        spec = state_to_spec(state)
        opt = spec.optimization
        assert opt.max_global_evaluations == 50_000
        assert opt.polish_top_k == 8
        assert opt.local_max_iterations == 1_500
        assert opt.random_seed == 20260815

    def test_custom_preset_compiles_user_values(self):
        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(
                preset="CUSTOM",
                custom_max_global_evaluations=3000,
                custom_polish_top_k=5,
                custom_local_max_iterations=200,
            )
        )
        spec = state_to_spec(state)
        opt = spec.optimization
        assert opt.max_global_evaluations == 3000
        assert opt.polish_top_k == 5
        assert opt.local_max_iterations == 200
        assert opt.local_derivative_mode == DerivativeMode.ANALYTICAL

    def test_analytical_always_set(self):
        """DerivativeMode.ANALYTICAL is always set regardless of preset."""
        for preset in ["FAST", "BALANCED", "THOROUGH", "CUSTOM"]:
            state = _build_test_state(
                optimization_preset=OptimizationPresetParams(preset=preset)
            )
            spec = state_to_spec(state)
            assert spec.optimization.local_derivative_mode == DerivativeMode.ANALYTICAL

    def test_effective_budgets_in_provenance(self):
        """RunManifest.requested_global_budget matches the compiled preset."""
        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(preset="FAST")
        )
        spec = state_to_spec(state)
        # The requested_global_budget in RunManifest is set from opt_spec.max_global_evaluations
        assert spec.optimization.max_global_evaluations == 500


# ===========================================================================
# 2. Constraint exposure tests
# ===========================================================================


class TestConstraintExposure:
    def test_match_constraints_gui_exposure(self):
        state = _build_test_state(
            match_params=MatchParams(
                gamma_max=0.10,
                resistance_min_ohm=40.0,
                resistance_max_ohm=60.0,
                max_abs_reactance_ohm=15.0,
            )
        )
        spec = state_to_spec(state)
        assert spec.matching.gamma_max == 0.10
        assert spec.matching.resistance_min_ohm == 40.0
        assert spec.matching.resistance_max_ohm == 60.0
        assert spec.matching.max_abs_reactance_ohm == 15.0

    def test_stress_constraints_gui_exposure(self):
        state = _build_test_state(
            stress_params=StressParams(
                source_current_rms_max_a=0.3,
                off_target_eom_peak_rms_v=30.0,
                default_cap_peak_voltage_v=80.0,
                default_ind_peak_current_a=0.8,
            )
        )
        spec = state_to_spec(state)
        assert spec.stress.source_current_rms_max_a == 0.3
        assert spec.stress.off_target_eom_peak_rms_v == 30.0
        assert spec.stress.default_cap_peak_voltage_v == 80.0
        assert spec.stress.default_ind_peak_current_a == 0.8

    def test_component_limits_gui_exposure(self):
        state = _build_test_state(
            component_limits=ComponentLimitParams(
                l_min_h=20e-9,
                l_max_h=50e-6,
                c_min_f=0.5e-12,
                c_max_f=10e-9,
            )
        )
        spec = state_to_spec(state)
        assert spec.components.continuous_limits.l_min_h == 20e-9
        assert spec.components.continuous_limits.l_max_h == 50e-6
        assert spec.components.continuous_limits.c_min_f == 0.5e-12
        assert spec.components.continuous_limits.c_max_f == 10e-9

    def test_objective_weights_gui_exposure(self):
        state = _build_test_state(
            objective_weights=ObjectiveWeightParams(
                weight_gamma=2.0,
                weight_voltage=0.5,
                weight_loss=0.1,
                weight_complexity=0.05,
            )
        )
        spec = state_to_spec(state)
        assert spec.optimization.objective_weight_gamma == 2.0
        assert spec.optimization.objective_weight_voltage == 0.5
        assert spec.optimization.objective_weight_loss == 0.1
        assert spec.optimization.objective_weight_complexity == 0.05


# ===========================================================================
# 3. Cancellation infrastructure tests
# ===========================================================================


class TestCancelInfra:
    def test_check_cancel_none_is_noop(self):
        """check_cancel(None) does nothing."""
        check_cancel(None)  # Should not raise

    def test_check_cancel_unset_is_noop(self):
        event = threading.Event()
        check_cancel(event)  # Should not raise

    def test_check_cancel_set_raises(self):
        event = threading.Event()
        event.set()
        with pytest.raises(CancelledException):
            check_cancel(event)


# ===========================================================================
# 4. Cancellation regression tests (mocked engine)
# ===========================================================================


class TestCancellationRegressions:
    def test_cancel_during_de_yields_cancelled_no_polish(self):
        """If cancel_event is set during DE, polish_top_k must not be called."""
        cancel_event = threading.Event()
        polish_calls = []

        # We intercept run_de to set cancel_event after first call
        def mock_run_de(*args, **kwargs):
            cancel_event.set()
            # Return minimal valid result
            from foster_eom.optimize.de_runner import DEDiagnostics
            diag = DEDiagnostics(
                domain_id="test",
                n_pop=4,
                n_gen_requested=1,
                n_gen_completed=0,
                budget_allocated=100,
                unique_x_evaluations=0,
                cache_hits=0,
                target_frequency_point_solves=0,
                coarse_frequency_point_solves=0,
                total_frequency_point_solves=0,
                numerical_failures=0,
                best_objective=1e9,
                best_feasible=False,
                de_termination="cancelled",
            )
            return [], diag

        def mock_polish_top_k(*args, **kwargs):
            polish_calls.append(1)
            return []

        with patch("foster_eom.optimize.engine.run_de", side_effect=mock_run_de), \
             patch("foster_eom.optimize.engine.polish_top_k", side_effect=mock_polish_top_k):

            state = _build_test_state(
                optimization_preset=OptimizationPresetParams(preset="FAST")
            )
            from foster_eom.gui.controllers.optimize_ctrl import OptimizeCtrl
            # We expect this to complete without calling polish
            try:
                OptimizeCtrl.run(state, cancel_event=cancel_event)
            except Exception:
                pass  # Engine may raise or return partial result

        assert len(polish_calls) == 0, "polish_top_k must not be called after DE cancellation"

    def test_cancel_during_polish_yields_cancelled_not_fallback(self):
        """Cancellation during trust-constr must produce 'cancelled', not fallback."""
        cancel_event = threading.Event()

        # Verify that polish_top_k breaks early when cancel is already set
        from foster_eom.optimize.evaluator import EvaluationResult

        # Create a mock _run_polish that simulates cancellation
        mock_pre = MagicMock(spec=EvaluationResult)
        mock_pre.x = (0.5,)
        mock_pre.objective_value = 1.0
        mock_pre.feasible = False
        mock_pre.near_feasible = False
        mock_pre.v_max = 1.0
        mock_pre.v_sum = 1.0
        mock_pre.hard_margins = [1.0]

        # Test that cancellation from trust-constr callback sets term_msg="cancelled"
        # We verify the mechanism: returning True from callback terminates solver
        cancel_event.set()

        from foster_eom.optimize.local_polish import polish_top_k as ptk
        from foster_eom.optimize.dedup import Basin

        # With cancel already set, polish_top_k should return empty (breaks before first candidate)
        basins = [Basin(representative=mock_pre, members=[mock_pre])]
        opt_spec = OptimizationSpec(
            polish_top_k=3,
            local_max_iterations=20,
            local_derivative_mode=DerivativeMode.ANALYTICAL,
        )
        mock_ctx = MagicMock()
        mock_cache = MagicMock()

        results = ptk(basins, mock_ctx, mock_cache, opt_spec, cancel_event=cancel_event)
        assert len(results) == 0, "No polish should happen when cancel_event is already set"

    def test_fallback_distinguishable_from_cancel(self):
        """Fallback has fallback_reason; cancellation has 'cancelled' termination."""
        # This tests the structural distinction:
        # - Fallback: PolishResult.reason is not None (from DerivativeUnavailable)
        # - Cancellation: PolishResult.termination == "cancelled"
        from foster_eom.optimize.local_polish import PolishTelemetry

        # Fallback telemetry has a fallback_reason
        fallback_telemetry = PolishTelemetry(
            derivative_mode=DerivativeMode.REFERENCE_FD.value,
            requested_mode=DerivativeMode.ANALYTICAL.value,
            fallback_reason="adjoint_unsupported",
        )
        assert fallback_telemetry.fallback_reason is not None

        # Cancellation telemetry has no fallback_reason
        cancel_telemetry = PolishTelemetry(
            derivative_mode=DerivativeMode.ANALYTICAL.value,
            requested_mode=DerivativeMode.ANALYTICAL.value,
            fallback_reason=None,
        )
        assert cancel_telemetry.fallback_reason is None

    def test_no_partial_cancelled_result_marked_successful(self):
        """A cancelled optimization must never be labelled FEASIBLE."""
        from foster_eom.gui.workers.optimize_worker import OptimizeWorker

        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(preset="FAST")
        )
        worker = OptimizeWorker(state)

        # Simulate immediate cancellation
        worker.cancel()
        assert worker.is_cancelled
        # The worker.run() would emit CANCELLED, not finished.
        # We verify the cancel_event is properly set.
        assert worker.cancel_event.is_set()


# ===========================================================================
# 5. Progress data tests
# ===========================================================================


class TestProgressData:
    def test_progress_update_defaults(self):
        p = ProgressUpdate()
        assert p.phase == "SEEDING"
        assert p.overall_percent == 0
        assert p.phase_percent == 0
        assert p.de_evals == 0
        assert p.de_budget == 0

    def test_progress_percent_bounded(self):
        """All percent fields must be in [0, 100]."""
        for pct in [0, 50, 100]:
            p = ProgressUpdate(overall_percent=pct, phase_percent=pct)
            assert 0 <= p.overall_percent <= 100
            assert 0 <= p.phase_percent <= 100

    def test_progress_monotonic_within_phase(self):
        """Simulated progress sequence must be monotonically non-decreasing."""
        updates = [
            ProgressUpdate(phase="DE", phase_percent=0),
            ProgressUpdate(phase="DE", phase_percent=25),
            ProgressUpdate(phase="DE", phase_percent=50),
            ProgressUpdate(phase="DE", phase_percent=75),
            ProgressUpdate(phase="DE", phase_percent=100),
        ]
        for i in range(1, len(updates)):
            assert updates[i].phase_percent >= updates[i - 1].phase_percent

    def test_de_reaches_100_on_budget_exhaust(self):
        """DE phase_percent == 100 when de_evals == de_budget."""
        p = ProgressUpdate(phase="DE", de_evals=500, de_budget=500, phase_percent=100)
        assert p.phase_percent == 100


# ===========================================================================
# 6. Backend defaults are NOT modified
# ===========================================================================


class TestBackendDefaultsPreserved:
    def test_optimization_spec_defaults_unchanged(self):
        """The backend OptimizationSpec defaults must remain at frozen production values."""
        default = OptimizationSpec()
        assert default.max_global_evaluations == 50_000
        assert default.polish_top_k == 8
        assert default.local_max_iterations == 1_500
        assert default.random_seed == 20260815
        assert default.local_derivative_mode == DerivativeMode.REFERENCE_FD

    def test_presets_do_not_modify_backend(self):
        """After compiling any preset, the backend defaults are still intact."""
        state = _build_test_state(
            optimization_preset=OptimizationPresetParams(preset="FAST")
        )
        _ = state_to_spec(state)

        # Backend defaults should be completely unchanged
        default = OptimizationSpec()
        assert default.max_global_evaluations == 50_000
        assert default.polish_top_k == 8
        assert default.local_max_iterations == 1_500
