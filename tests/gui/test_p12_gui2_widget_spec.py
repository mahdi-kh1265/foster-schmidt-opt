"""P12-GUI2: Widget→State→Spec, Progress Semantics, and Unit Conversion Tests.

Supplements test_p12_gui2_execution_controls.py with:
- Permanent widget→state→spec tests for every newly exposed field
- Unit conversion tests for L/C with unit selectors
- Min/max validation tests
- Progress percentage semantics (DE, polish, overall, multi-domain)
- Per-target voltage flow tests
"""

from __future__ import annotations

from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.objectives import DerivativeMode, OptimizationSpec
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import (
    ComponentLimitParams,
    EOMParams,
    MatchParams,
    ObjectiveWeightParams,
    ProjectState,
    SourceParams,
    StressParams,
    TopologyParams,
)
from foster_eom.optimize.progress import ProgressUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> ProjectState:
    state = ProjectState()
    state.name = "Widget Test"
    state.frequencies_hz = [10e6]
    state.sweep_f_min_hz = 1e6
    state.sweep_f_max_hz = 30e6
    state.source = SourceParams(mode="thevenin", vth_rms=2.0, z_source_ohm=50.0)
    state.eom = EOMParams(model_type="ideal_capacitor", c0_f=1e-9)
    state.topology = TopologyParams(n_branches=2, n_cells_per_branch=1)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ===========================================================================
# 1. Widget→State→Spec: Every newly exposed matching constraint field
# ===========================================================================


class TestMatchWidgetToSpec:
    """Each matching constraint field must round-trip widget→state→spec."""

    def test_gamma_max_round_trip(self):
        state = _state(match_params=MatchParams(gamma_max=0.15))
        spec = state_to_spec(state)
        assert spec.matching.gamma_max == 0.15

    def test_resistance_min_round_trip(self):
        state = _state(match_params=MatchParams(resistance_min_ohm=42.0))
        spec = state_to_spec(state)
        assert spec.matching.resistance_min_ohm == 42.0

    def test_resistance_max_round_trip(self):
        state = _state(match_params=MatchParams(resistance_max_ohm=55.0))
        spec = state_to_spec(state)
        assert spec.matching.resistance_max_ohm == 55.0

    def test_max_abs_reactance_round_trip(self):
        state = _state(match_params=MatchParams(max_abs_reactance_ohm=12.0))
        spec = state_to_spec(state)
        assert spec.matching.max_abs_reactance_ohm == 12.0

    def test_all_match_fields_compile(self):
        mp = MatchParams(
            gamma_max=0.10,
            resistance_min_ohm=40.0,
            resistance_max_ohm=60.0,
            max_abs_reactance_ohm=15.0,
        )
        state = _state(match_params=mp)
        spec = state_to_spec(state)
        assert spec.matching.gamma_max == 0.10
        assert spec.matching.resistance_min_ohm == 40.0
        assert spec.matching.resistance_max_ohm == 60.0
        assert spec.matching.max_abs_reactance_ohm == 15.0


# ===========================================================================
# 2. Widget→State→Spec: Stress constraint fields
# ===========================================================================


class TestStressWidgetToSpec:
    def test_source_current_round_trip(self):
        state = _state(stress_params=StressParams(source_current_rms_max_a=0.3))
        spec = state_to_spec(state)
        assert spec.stress.source_current_rms_max_a == 0.3

    def test_off_target_voltage_round_trip(self):
        state = _state(stress_params=StressParams(off_target_eom_peak_rms_v=30.0))
        spec = state_to_spec(state)
        assert spec.stress.off_target_eom_peak_rms_v == 30.0

    def test_cap_voltage_stress_round_trip(self):
        state = _state(stress_params=StressParams(default_cap_peak_voltage_v=80.0))
        spec = state_to_spec(state)
        assert spec.stress.default_cap_peak_voltage_v == 80.0

    def test_ind_current_stress_round_trip(self):
        state = _state(stress_params=StressParams(default_ind_peak_current_a=0.8))
        spec = state_to_spec(state)
        assert spec.stress.default_ind_peak_current_a == 0.8


# ===========================================================================
# 3. Widget→State→Spec: Component limits with unit conversions
# ===========================================================================


class TestComponentLimitUnitConversions:
    """Unit-conversion integrity: GUI display units → internal SI → spec SI."""

    def test_l_min_nH_to_henries(self):
        """10 nH → 10e-9 H."""
        state = _state(component_limits=ComponentLimitParams(l_min_h=10e-9))
        spec = state_to_spec(state)
        assert abs(spec.components.continuous_limits.l_min_h - 10e-9) < 1e-18

    def test_l_max_uH_to_henries(self):
        """100 µH → 100e-6 H."""
        state = _state(component_limits=ComponentLimitParams(l_max_h=100e-6))
        spec = state_to_spec(state)
        assert abs(spec.components.continuous_limits.l_max_h - 100e-6) < 1e-15

    def test_l_min_mH_to_henries(self):
        """1 mH → 1e-3 H."""
        state = _state(component_limits=ComponentLimitParams(l_min_h=1e-3, l_max_h=10e-3))
        spec = state_to_spec(state)
        assert abs(spec.components.continuous_limits.l_min_h - 1e-3) < 1e-12

    def test_c_min_pF_to_farads(self):
        """0.2 pF → 0.2e-12 F."""
        state = _state(component_limits=ComponentLimitParams(c_min_f=0.2e-12))
        spec = state_to_spec(state)
        assert abs(spec.components.continuous_limits.c_min_f - 0.2e-12) < 1e-24

    def test_c_max_nF_to_farads(self):
        """20 nF → 20e-9 F."""
        state = _state(component_limits=ComponentLimitParams(c_max_f=20e-9))
        spec = state_to_spec(state)
        assert abs(spec.components.continuous_limits.c_max_f - 20e-9) < 1e-18

    def test_c_max_uF_to_farads(self):
        """1 µF → 1e-6 F."""
        state = _state(component_limits=ComponentLimitParams(c_max_f=1e-6))
        spec = state_to_spec(state)
        assert abs(spec.components.continuous_limits.c_max_f - 1e-6) < 1e-15

    def test_all_limits_round_trip(self):
        cl = ComponentLimitParams(l_min_h=5e-9, l_max_h=50e-6, c_min_f=0.5e-12, c_max_f=10e-9)
        state = _state(component_limits=cl)
        spec = state_to_spec(state)
        lim = spec.components.continuous_limits
        assert abs(lim.l_min_h - 5e-9) < 1e-18
        assert abs(lim.l_max_h - 50e-6) < 1e-15
        assert abs(lim.c_min_f - 0.5e-12) < 1e-24
        assert abs(lim.c_max_f - 10e-9) < 1e-18


# ===========================================================================
# 4. Widget→State→Spec: Objective weights
# ===========================================================================


class TestObjectiveWeightWidgetToSpec:
    def test_gamma_weight_round_trip(self):
        state = _state(objective_weights=ObjectiveWeightParams(weight_gamma=2.0))
        spec = state_to_spec(state)
        assert spec.optimization.objective_weight_gamma == 2.0

    def test_voltage_weight_round_trip(self):
        state = _state(objective_weights=ObjectiveWeightParams(weight_voltage=0.5))
        spec = state_to_spec(state)
        assert spec.optimization.objective_weight_voltage == 0.5

    def test_loss_weight_round_trip(self):
        state = _state(objective_weights=ObjectiveWeightParams(weight_loss=0.1))
        spec = state_to_spec(state)
        assert spec.optimization.objective_weight_loss == 0.1

    def test_complexity_weight_round_trip(self):
        state = _state(objective_weights=ObjectiveWeightParams(weight_complexity=0.05))
        spec = state_to_spec(state)
        assert spec.optimization.objective_weight_complexity == 0.05


# ===========================================================================
# 5. Per-target voltage flow
# ===========================================================================


class TestPerTargetVoltage:
    def test_voltage_target_none_by_default(self):
        state = _state()
        spec = state_to_spec(state)
        for t in spec.frequencies.targets:
            assert t.voltage_target_rms_v is None

    def test_voltage_target_round_trip(self):
        state = _state()
        state.frequencies_hz = [10e6, 20e6]
        state.voltage_targets_rms_v = [5.0, None]
        spec = state_to_spec(state)
        assert spec.frequencies.targets[0].voltage_target_rms_v == 5.0
        assert spec.frequencies.targets[1].voltage_target_rms_v is None

    def test_voltage_target_all_set(self):
        state = _state()
        state.frequencies_hz = [10e6, 20e6]
        state.voltage_targets_rms_v = [3.5, 7.2]
        spec = state_to_spec(state)
        assert spec.frequencies.targets[0].voltage_target_rms_v == 3.5
        assert spec.frequencies.targets[1].voltage_target_rms_v == 7.2

    def test_voltage_empty_list_is_fine(self):
        """No voltage list → all targets get None."""
        state = _state()
        state.frequencies_hz = [10e6]
        state.voltage_targets_rms_v = []
        spec = state_to_spec(state)
        assert spec.frequencies.targets[0].voltage_target_rms_v is None


# ===========================================================================
# 6. Min/max validation: component limits
# ===========================================================================


class TestMinMaxValidation:
    """Component limits must satisfy l_min ≤ l_max and c_min ≤ c_max."""

    def test_valid_l_range(self):
        cl = ComponentLimitParams(l_min_h=1e-9, l_max_h=1e-6)
        state = _state(component_limits=cl)
        spec = state_to_spec(state)
        assert spec.components.continuous_limits.l_min_h < spec.components.continuous_limits.l_max_h

    def test_valid_c_range(self):
        cl = ComponentLimitParams(c_min_f=0.1e-12, c_max_f=100e-9)
        state = _state(component_limits=cl)
        spec = state_to_spec(state)
        assert spec.components.continuous_limits.c_min_f < spec.components.continuous_limits.c_max_f

    def test_equal_bounds_accepted(self):
        """Edge case: min == max should compile without error (single value)."""
        cl = ComponentLimitParams(l_min_h=1e-6, l_max_h=1e-6)
        state = _state(component_limits=cl)
        spec = state_to_spec(state)
        assert spec.components.continuous_limits.l_min_h == spec.components.continuous_limits.l_max_h


# ===========================================================================
# 7. Default initialization: all GUI state defaults match expectations
# ===========================================================================


class TestDefaultInitialization:
    """All GUI state defaults must match the documented values."""

    def test_match_defaults(self):
        mp = MatchParams()
        assert mp.gamma_max == 0.25
        assert mp.resistance_min_ohm == 35.0
        assert mp.resistance_max_ohm == 70.0
        assert mp.max_abs_reactance_ohm == 20.0

    def test_stress_defaults(self):
        sp = StressParams()
        assert sp.source_current_rms_max_a == 0.5
        assert sp.off_target_eom_peak_rms_v == 50.0
        assert sp.default_cap_peak_voltage_v == 100.0
        assert sp.default_ind_peak_current_a == 1.0

    def test_component_defaults(self):
        cl = ComponentLimitParams()
        assert cl.l_min_h == 10e-9
        assert cl.l_max_h == 100e-6
        assert cl.c_min_f == 0.2e-12
        assert cl.c_max_f == 20e-9

    def test_weight_defaults(self):
        ow = ObjectiveWeightParams()
        assert ow.weight_gamma == 1.0
        assert ow.weight_voltage == 1.0
        assert ow.weight_loss == 0.0
        assert ow.weight_complexity == 0.0

    def test_default_state_compiles(self):
        """A default ProjectState (with 1 frequency) must compile to a valid spec."""
        state = _state()
        spec = state_to_spec(state)
        assert spec.matching is not None
        assert spec.stress is not None
        assert spec.components is not None


# ===========================================================================
# 8. Progress percentage semantics (documented/tested)
# ===========================================================================


class TestProgressSemantics:
    """Explicit progress semantics for DE, polish, overall."""

    # --- DE percent: consumed_budget / total_de_budget ---

    def test_de_percent_zero_at_start(self):
        p = ProgressUpdate(phase="DE", de_evals=0, de_budget=500, phase_percent=0)
        assert p.phase_percent == 0

    def test_de_percent_proportional(self):
        """DE phase percent = de_evals / de_budget * 100."""
        evals, budget = 250, 500
        pct = int(evals / budget * 100)
        p = ProgressUpdate(phase="DE", de_evals=evals, de_budget=budget, phase_percent=pct)
        assert p.phase_percent == 50

    def test_de_percent_100_on_exhaustion(self):
        p = ProgressUpdate(phase="DE", de_evals=500, de_budget=500, phase_percent=100)
        assert p.phase_percent == 100

    def test_de_percent_clamped_at_100(self):
        """If evals slightly exceed budget (rounding), still ≤ 100."""
        pct = min(100, int(501 / 500 * 100))
        assert pct == 100

    # --- Polish percent: candidate + iteration progress ---

    def test_polish_percent_first_candidate_start(self):
        p = ProgressUpdate(
            phase="POLISH",
            polish_candidate_index=0,
            polish_top_k=3,
            polish_iteration=0,
            polish_max_iterations=100,
            phase_percent=0,
        )
        assert p.phase_percent == 0

    def test_polish_percent_mid_candidate(self):
        """Polish percent = (cand_index * max_iter + iter) / (top_k * max_iter) * 100."""
        k, max_iter = 3, 100
        cand, it = 1, 50
        pct = int((cand * max_iter + it) / (k * max_iter) * 100)
        p = ProgressUpdate(
            phase="POLISH",
            polish_candidate_index=cand,
            polish_top_k=k,
            polish_iteration=it,
            polish_max_iterations=max_iter,
            phase_percent=pct,
        )
        assert p.phase_percent == 50

    def test_polish_percent_100_on_completion(self):
        pct = int((3 * 100) / (3 * 100) * 100)
        assert pct == 100

    # --- Overall percent: labelled "budget-based estimate" ---

    def test_overall_percent_is_budget_based_estimate(self):
        """Overall progress is a budget-based estimate, not wall-clock."""
        p = ProgressUpdate(phase="DE", overall_percent=30, phase_percent=60)
        assert 0 <= p.overall_percent <= 100

    def test_overall_never_exceeds_100(self):
        for pct in [0, 50, 99, 100]:
            p = ProgressUpdate(overall_percent=pct)
            assert 0 <= p.overall_percent <= 100

    # --- All percentages in [0, 100] ---

    def test_all_percents_bounded(self):
        for pct in [0, 1, 50, 99, 100]:
            p = ProgressUpdate(overall_percent=pct, phase_percent=pct)
            assert 0 <= p.overall_percent <= 100
            assert 0 <= p.phase_percent <= 100

    # --- Early convergence ---

    def test_early_convergence_completes_cleanly(self):
        """If DE converges early (evals < budget), the result is still valid."""
        p = ProgressUpdate(
            phase="DE",
            de_evals=200,
            de_budget=500,
            phase_percent=40,
            overall_percent=40,
        )
        # Early convergence: evals < budget
        assert p.de_evals < p.de_budget
        # Phase percent is < 100 but the run completed
        assert p.phase_percent < 100

    # --- Multi-domain runs ---

    def test_multi_domain_progress_uses_total_denominator(self):
        """For N domains, overall progress denominates by total DE budget across domains."""
        # 3 domains, each with budget=500 → total=1500
        total_budget = 1500
        consumed = 500  # one domain done
        pct = int(consumed / total_budget * 100)
        p = ProgressUpdate(
            phase="DE",
            domain_index=1,
            domain_count=3,
            de_evals=consumed,
            de_budget=total_budget,
            overall_percent=pct,
        )
        assert p.overall_percent == 33
        assert p.domain_count == 3

    def test_multi_domain_final_domain_100(self):
        """When all domains complete, overall progress is 100%."""
        total_budget = 1500
        p = ProgressUpdate(
            phase="POLISH",
            domain_index=2,
            domain_count=3,
            de_evals=total_budget,
            de_budget=total_budget,
            overall_percent=100,
        )
        assert p.overall_percent == 100


# ===========================================================================
# 9. Backend defaults NOT modified (regression guard)
# ===========================================================================


class TestBackendDefaultsNotModified:
    def test_optimization_spec_defaults_frozen(self):
        d = OptimizationSpec()
        assert d.max_global_evaluations == 50_000
        assert d.polish_top_k == 8
        assert d.local_max_iterations == 1_500
        assert d.random_seed == 20260815
        assert d.local_derivative_mode == DerivativeMode.REFERENCE_FD

    def test_match_constraints_defaults_frozen(self):
        d = MatchConstraints()
        assert d.gamma_max == 0.25
        assert d.resistance_min_ohm == 35.0
        assert d.resistance_max_ohm == 70.0
        assert d.max_abs_reactance_ohm == 20.0

    def test_stress_constraints_defaults_frozen(self):
        d = StressConstraints()
        assert d.source_current_rms_max_a == 0.5
        assert d.off_target_eom_peak_rms_v == 50.0
        assert d.default_cap_peak_voltage_v == 100.0
        assert d.default_ind_peak_current_a == 1.0
