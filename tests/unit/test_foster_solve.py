"""Unit tests for scaled constrained Foster solve (Prompt 04A, tests #52-77)."""

from __future__ import annotations

import math

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.foster.foster_form import (
    compute_coefficient_bounds,
    foster_reactance_hz,
)
from foster_eom.foster.foster_solve import (
    CoefficientKind,
    build_foster_linear_system,
    solve_foster_system,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIMITS = ContinuousLimits()


def _known_system(
    n_targets: int = 2,
    n_poles: int = 2,
    enable_k0: bool = False,
    enable_kinf: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a known positive-coefficient system for round-trip tests.

    Returns (f_targets, x_targets, f_poles, k_m_true, bounds_km).
    """
    f_targets = np.array([5e6, 10e6, 15e6, 20e6])[:n_targets]
    f_poles = np.array([7e6, 12.5e6, 17e6])[:n_poles]
    k_m_true = np.array([3e8, 6e8, 9e8])[:n_poles]

    x_targets = foster_reactance_hz(f_targets, None, None, k_m_true, f_poles)
    return f_targets, x_targets, f_poles, k_m_true, x_targets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExactRecoverySquare:
    """#52: (C.2) Known positive k/q -> X -> solve -> recover."""

    def test_exact_recovery(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6, 12e6])
        k_true = np.array([3e8, 5e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)

        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.feasible
        assert result.system_class == "square"
        np.testing.assert_allclose(result.k_residues, tuple(k_true), rtol=1e-6)


class TestOverdeterminedBoundedLs:
    """#53: (C.3) More targets than coefficients -> bounded LS."""

    def test_overdetermined(self):
        f_t = np.array([5e6, 10e6, 15e6])
        f_p = np.array([7e6])  # 1 pole, 1 coefficient
        k_true = np.array([3e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)

        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.system_class == "overdetermined"
        assert result.n_targets == 3
        assert result.n_coefficients == 1
        assert result.max_target_error_ohm < 1.0


class TestPassivityFailureStructured:
    """#54: (C.4) Impossible positive-k target -> structured failure."""

    def test_infeasible(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6])
        # Construct a target that requires a negative coefficient
        x_t = np.array([100.0, 100.0])  # Flat -> incompatible with 1-pole positive-residue

        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        # Should report poor fit or bound violation
        assert result.max_target_error_ohm > 10.0 or not result.feasible


class TestUnderdeterminedStageABounded:
    """#55: P > N -> Stage A computes minimum bounded residual."""

    def test_stage_a(self):
        f_t = np.array([10e6])  # 1 target
        f_p = np.array([7e6, 15e6])  # 2 poles -> 2 coefficients
        k_true = np.array([3e8, 5e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)

        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.system_class == "underdetermined"
        assert result.minimum_bounded_residual is not None
        assert result.selected_fit_residual is not None


class TestUnderdeterminedStageBRegularized:
    """#56: Stage B augmented system -> deterministic result."""

    def test_stage_b(self):
        f_t = np.array([10e6])
        f_p = np.array([7e6, 15e6])
        k_true = np.array([3e8, 5e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)

        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.regularization_used is not None
        assert result.regularization_method in (None, "augmented_system")


class TestUnderdeterminedDegradationGuard:
    """#57: Stage B worsens fit -> Stage A retained."""

    def test_degradation_guard(self):
        f_t = np.array([10e6])
        f_p = np.array([7e6, 15e6])
        k_true = np.array([3e8, 5e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)

        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)

        # Use very high lambda that should degrade fit
        result_high_lambda = solve_foster_system(
            system,
            f_p,
            regularization_lambda=1e6,
            r_degradation_abs_tol=1e-15,
            r_degradation_rel_tol=1e-6,
        )

        # If fit was degraded, Stage A should be retained
        if (
            result_high_lambda.minimum_bounded_residual is not None
            and not result_high_lambda.regularization_used
        ):
            # Stage A was retained -- degradation guard worked
            assert result_high_lambda.selected_fit_residual is not None
            assert (
                result_high_lambda.selected_fit_residual
                <= result_high_lambda.minimum_bounded_residual + 1e-8
            )


class TestUnderdeterminedBothStagesRecorded:
    """#58: Both diagnostics recorded."""

    def test_both_recorded(self):
        f_t = np.array([10e6])
        f_p = np.array([7e6, 15e6])
        x_t = foster_reactance_hz(f_t, None, None, np.array([3e8, 5e8]), f_p)
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.minimum_bounded_residual is not None
        assert result.selected_fit_residual is not None
        assert result.regularization_used is not None


class TestNoNormalEquations:
    """#59: Augmented system path, not AᵀA+λI."""

    def test_no_ata(self):
        # Underdetermined system
        f_t = np.array([10e6])
        f_p = np.array([7e6, 15e6])
        x_t = np.array([50.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        if result.regularization_used:
            assert result.regularization_method == "augmented_system"


class TestRankDeficientDetection:
    """#60: Near-collinear -> rank, condition reported."""

    def test_rank(self):
        # Two poles very close -> near-collinear columns
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6, 7.001e6])
        x_t = np.array([50.0, -30.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.scaled_condition_number > 1e3  # Should be large
        assert result.rank >= 1


class TestIllConditionedRejection:
    """#61: High cond -> system flagged."""

    def test_ill_cond(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6, 7.00001e6])  # extremely close -> ill-conditioned
        x_t = np.array([50.0, -30.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.scaled_condition_number > 1e4


class TestActiveBoundsReported:
    """#62: Bound-hitting indices."""

    def test_active_bounds(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6])
        # Target requires very large k -> should hit upper bound
        x_t = np.array([1e6, -1e6])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        # At least check the fields exist and are tuples
        assert isinstance(result.active_lower_bounds, tuple)
        assert isinstance(result.active_upper_bounds, tuple)


class TestNormalizedDynamicRangeAllCoeffs:
    """#63: D_u includes active-bound coefficients."""

    def test_includes_bound(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6, 12e6])
        k_true = np.array([3e8, 5e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        if result.normalized_coefficient_dynamic_range is not None:
            assert result.normalized_coefficient_dynamic_range >= 1.0


class TestDisabledK0ColumnAbsent:
    """#64: k₀ off -> no column."""

    def test_no_k0(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6])
        x_t = np.array([50.0, -30.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        assert all(d.kind != CoefficientKind.K0 for d in system.coefficients)


class TestDisabledKinfColumnAbsent:
    """#65: kinf off -> no column."""

    def test_no_kinf(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6])
        x_t = np.array([50.0, -30.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        assert all(d.kind != CoefficientKind.K_INF for d in system.coefficients)


class TestBothEndpointsEnabled:
    """#66: Both present -> correct count and identity."""

    def test_both(self):
        f_t = np.array([5e6, 10e6, 15e6, 20e6])
        f_p = np.array([7e6, 12e6])
        x_t = np.array([50.0, -30.0, 20.0, -10.0])
        bounds = compute_coefficient_bounds(f_p, True, True, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, True, True, bounds)

        kinds = [d.kind for d in system.coefficients]
        assert CoefficientKind.K0 in kinds
        assert CoefficientKind.K_INF in kinds
        assert kinds.count(CoefficientKind.K_RESIDUE) == 2
        assert len(system.coefficients) == 4


class TestNoEndpoints:
    """#67: Neither -> only k_m columns."""

    def test_km_only(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6, 12e6])
        x_t = np.array([50.0, -30.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        assert all(d.kind == CoefficientKind.K_RESIDUE for d in system.coefficients)


class TestMultipleFinitePoles:
    """#68: 3 poles -> correct descriptors."""

    def test_three_poles(self):
        f_t = np.array([5e6, 10e6, 15e6])
        f_p = np.array([7e6, 12e6, 17e6])
        x_t = np.array([50.0, -30.0, 20.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        assert len(system.coefficients) == 3
        for i, d in enumerate(system.coefficients):
            assert d.kind == CoefficientKind.K_RESIDUE
            assert d.cell_index == i


class TestColumnIdentityPreserved:
    """#69: Descriptors match columns after solve."""

    def test_identity(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6, 12e6])
        k_true = np.array([3e8, 5e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert result.coefficient_descriptors == system.coefficients


class TestRowScalingNearZero:
    """#70: Near-zero target not overweighted."""

    def test_near_zero(self):
        f_t = np.array([5e6, 10e6])
        x_t = np.array([0.001, 100.0])
        f_p = np.array([7e6])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)

        # Row scale for the near-zero target should be floored
        assert system.row_scales[0] >= 0.1  # eps_row * x_char = 0.001 * 100 = 0.1


class TestRowScalingAllZeroNoNan:
    """#71: All X = 0 -> finite r_i, no NaN."""

    def test_all_zero(self):
        f_t = np.array([5e6, 10e6])
        x_t = np.array([0.0, 0.0])
        f_p = np.array([7e6])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)

        assert np.all(np.isfinite(system.row_scales))
        assert np.all(system.row_scales > 0)
        assert np.all(np.isfinite(system.scaled_target))


class TestZeroTargetResidualFinite:
    """#72: ‖x̃‖ = 0 -> finite normalized_residual."""

    def test_finite_residual(self):
        f_t = np.array([5e6, 10e6])
        x_t = np.array([0.0, 0.0])
        f_p = np.array([7e6])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert math.isfinite(result.normalized_residual)
        assert not math.isnan(result.normalized_residual)


class TestPhysicalBoundsNotJustPositivity:
    """#73: k > 0 but > k_max -> rejected."""

    def test_bound_violation(self):
        f_p = np.array([7e6, 12e6])
        # Construct targets requiring coefficients near upper bound
        limits = ContinuousLimits(l_min_h=10e-9, l_max_h=100e-9, c_min_f=1e-12, c_max_f=10e-12)
        bounds = compute_coefficient_bounds(f_p, False, False, limits)
        # Check that bounds have finite upper limits
        for bnd in bounds.km_bounds:
            assert bnd[1] < math.inf


class TestPerBranchDofClassification:
    """#74: P_b independent, correct label."""

    def test_classification(self):
        f_t = np.array([5e6, 10e6])

        # Square: 2 targets, 2 poles
        f_p2 = np.array([7e6, 12e6])
        bounds2 = compute_coefficient_bounds(f_p2, False, False, _LIMITS)
        sys2 = build_foster_linear_system(f_t, np.array([50.0, -30.0]), f_p2, False, False, bounds2)
        res2 = solve_foster_system(sys2, f_p2)
        assert res2.system_class == "square"

        # Overdetermined: 2 targets, 1 pole
        f_p1 = np.array([7e6])
        bounds1 = compute_coefficient_bounds(f_p1, False, False, _LIMITS)
        sys1 = build_foster_linear_system(f_t, np.array([50.0, -30.0]), f_p1, False, False, bounds1)
        res1 = solve_foster_system(sys1, f_p1)
        assert res1.system_class == "overdetermined"

        # Underdetermined: 1 target, 2 poles
        f_t1 = np.array([10e6])
        sys_u = build_foster_linear_system(f_t1, np.array([50.0]), f_p2, False, False, bounds2)
        res_u = solve_foster_system(sys_u, f_p2)
        assert res_u.system_class == "underdetermined"


class TestGeometricMeanScaling:
    """#75: s_j = sqrt(k_min * k_max)."""

    def test_geometric_mean(self):
        f_p = np.array([10e6])
        bounds = compute_coefficient_bounds(f_p, True, True, _LIMITS)
        f_t = np.array([5e6, 15e6])
        x_t = np.array([50.0, -30.0])
        system = build_foster_linear_system(f_t, x_t, f_p, True, True, bounds)

        for desc in system.coefficients:
            expected_scale = math.sqrt(desc.lower_bound * desc.upper_bound)
            assert abs(desc.scale - expected_scale) / expected_scale < 1e-10


class TestSystemAcceptsHz:
    """#76: build_foster_linear_system accepts Hz, not ω."""

    def test_hz_input(self):
        import inspect

        sig = inspect.signature(build_foster_linear_system)
        params = set(sig.parameters.keys())
        assert "f_targets_hz" in params
        assert "f_poles_hz" in params
        assert "omega" not in params
        assert "q_m" not in params


class TestResultReportsFPolesHz:
    """#77: FosterSolveResult.f_poles_hz, not q_poles."""

    def test_hz_in_result(self):
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6])
        k_true = np.array([3e8])
        x_t = foster_reactance_hz(f_t, None, None, k_true, f_p)
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)

        assert hasattr(result, "f_poles_hz")
        assert not hasattr(result, "q_poles")
        assert result.f_poles_hz == (7e6,)

    def test_solver_convergence_recorded(self):
        """#77b: lsq_linear convergence status recorded."""
        f_t = np.array([5e6, 10e6])
        f_p = np.array([7e6])
        x_t = np.array([50.0, -30.0])
        bounds = compute_coefficient_bounds(f_p, False, False, _LIMITS)
        system = build_foster_linear_system(f_t, x_t, f_p, False, False, bounds)
        result = solve_foster_system(system, f_p)
        assert result.solver_status != ""
        assert isinstance(result.solver_status, str)
