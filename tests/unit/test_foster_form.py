"""Unit tests for Foster-form mathematics (Prompt 04A, tests #38-51)."""

from __future__ import annotations

import math

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.foster.foster_form import (
    _TWO_PI,
    _foster_reactance_omega,
    coefficients_to_components,
    compute_coefficient_bounds,
    find_required_pole_intervals,
    foster_derivative_hz,
    foster_reactance_hz,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIMITS = ContinuousLimits()


class TestFosterEvalKnown:
    """#38: Known coefficients -> X(ω) matches manual computation."""

    def test_known(self):
        # X(ω) = -k0/ω + ω*k_inf + ω*k1/(q1 - ω²)
        k0 = 1e9  # 1/C0
        k_inf = 1e-6  # L_inf
        k_m = np.array([5e8])
        f_poles = np.array([8e6])
        f_eval = np.array([5e6])

        omega = _TWO_PI * 5e6
        q1 = (_TWO_PI * 8e6) ** 2

        expected = -k0 / omega + omega * k_inf + omega * 5e8 / (q1 - omega**2)
        result = foster_reactance_hz(f_eval, k0, k_inf, k_m, f_poles)
        assert abs(result[0] - expected) < 1e-6


class TestFosterDerivativePositive:
    """#39: Positive coefficients -> dX/dω > 0 between poles."""

    def test_positive(self):
        k0 = 1e9
        k_inf = 1e-7
        k_m = np.array([2e8, 4e8])
        f_poles = np.array([5e6, 15e6])

        # Evaluate between poles: at 3e6, 10e6, 20e6
        f_eval = np.array([3e6, 10e6, 20e6])
        dxdf = foster_derivative_hz(f_eval, k0, k_inf, k_m, f_poles)
        assert np.all(dxdf > 0), f"dX/df should be positive, got {dxdf}"


class TestFosterDerivativeHzFiniteDifference:
    """#39b: Numerical finite-difference test for dX/df magnitude."""

    def test_finite_difference(self):
        k0 = 5e8
        k_inf = 2e-7
        k_m = np.array([1e9])
        f_poles = np.array([10e6])

        f0 = 6e6
        df = 1.0  # 1 Hz step
        x_plus = foster_reactance_hz(np.array([f0 + df]), k0, k_inf, k_m, f_poles)[0]
        x_minus = foster_reactance_hz(np.array([f0 - df]), k0, k_inf, k_m, f_poles)[0]
        numerical_dxdf = (x_plus - x_minus) / (2 * df)

        analytic_dxdf = foster_derivative_hz(np.array([f0]), k0, k_inf, k_m, f_poles)[0]
        rel_err = abs(analytic_dxdf - numerical_dxdf) / abs(numerical_dxdf)
        assert rel_err < 1e-6, (
            f"Analytic dX/df = {analytic_dxdf:.6e}, "
            f"numerical = {numerical_dxdf:.6e}, "
            f"rel err = {rel_err:.2e}"
        )


class TestFosterPoleTransition:
    """#40: +inf -> -inf across each finite pole."""

    def test_transition(self):
        k_m = np.array([1e9])
        f_poles = np.array([10e6])

        # Just below pole (10 Hz away)
        f_below = np.array([10e6 - 10])
        x_below = foster_reactance_hz(f_below, None, None, k_m, f_poles)
        assert x_below[0] > 1e6  # very large positive

        # Just above pole (10 Hz away)
        f_above = np.array([10e6 + 10])
        x_above = foster_reactance_hz(f_above, None, None, k_m, f_poles)
        assert x_above[0] < -1e6  # very large negative


class TestMonotonicZeroCrossingNoPole:
    """#41: X_i < 0, X_{i+1} > 0, increasing -> no required pole."""

    def test_zero_crossing(self):
        f = np.array([5e6, 10e6, 15e6])
        x = np.array([-100.0, 0.0, 200.0])  # monotonically increasing
        intervals = find_required_pole_intervals(f, x)
        assert len(intervals) == 0


class TestDecreasingTargetsRequirePole:
    """#42: X_{i+1} < X_i -> required interval."""

    def test_decreasing(self):
        f = np.array([5e6, 10e6, 15e6])
        x = np.array([100.0, -50.0, 200.0])
        intervals = find_required_pole_intervals(f, x)
        assert len(intervals) == 1
        assert intervals[0].f_lo_hz == 5e6
        assert intervals[0].f_hi_hz == 10e6


class TestTrivialZeroTargetNoPoles:
    """#43: [0, 0, 0] -> ZERO_IMPEDANCE path, no pole requirements."""

    def test_trivial_zero(self):
        f = np.array([5e6, 10e6, 15e6])
        x = np.array([0.0, 0.0, 0.0])
        intervals = find_required_pole_intervals(f, x)
        assert len(intervals) == 0


class TestNonzeroFlatTargetRequiresPoles:
    """#44: [10, 10, 10] -> not trivial zero, requires poles.

    Equal values trigger X_{i+1} <= X_i, so a pole is required
    in each interval for a nontrivial positive-residue Foster.
    """

    def test_flat_nonzero(self):
        f = np.array([5e6, 10e6, 15e6])
        x = np.array([10.0, 10.0, 10.0])
        intervals = find_required_pole_intervals(f, x)
        # X[1] <= X[0] and X[2] <= X[1] -> 2 required intervals
        assert len(intervals) == 2


class TestRequiredIntervalsInHz:
    """#45: Output intervals are in Hz."""

    def test_hz_output(self):
        f = np.array([5e6, 10e6])
        x = np.array([100.0, -50.0])
        intervals = find_required_pole_intervals(f, x)
        assert len(intervals) == 1
        # Check they are in MHz range, not rad/s or q
        assert 1e6 < intervals[0].f_lo_hz < 1e8
        assert 1e6 < intervals[0].f_hi_hz < 1e8


class TestCoefficientBoundsAcceptsHz:
    """#46: compute_coefficient_bounds accepts Hz."""

    def test_hz_input(self):
        f_poles = np.array([10e6])
        bounds = compute_coefficient_bounds(f_poles, True, True, _LIMITS)
        assert bounds.k0_bounds is not None
        assert bounds.kinf_bounds is not None
        assert len(bounds.km_bounds) == 1


class TestCoefficientBoundsInfeasibleCell:
    """#47: k_m,min > k_m,max -> detected."""

    def test_infeasible(self):
        # Very high pole freq -> q_m large -> k_m,min = q_m*L_min >> 1/C_min
        f_poles = np.array([1e12])  # 1 THz
        bounds = compute_coefficient_bounds(f_poles, False, False, _LIMITS)
        assert bounds.any_infeasible
        assert 0 in bounds.infeasible_cells


class TestComponentConversionHzInput:
    """#48: coefficients_to_components accepts f_poles_hz."""

    def test_conversion(self):
        k_m = np.array([1e9])
        f_poles = np.array([10e6])
        comp = coefficients_to_components(None, None, k_m, f_poles)
        assert comp.c0_f is None
        assert comp.l_inf_h is None
        assert len(comp.cells) == 1
        # C_m = 1/k_m = 1e-9 F
        assert abs(comp.cells[0].c_f - 1e-9) < 1e-15
        # L_m = k_m/q_m
        q_m = (_TWO_PI * 10e6) ** 2
        expected_l = 1e9 / q_m
        assert abs(comp.cells[0].l_h - expected_l) / expected_l < 1e-10
        # f_pole reported in Hz
        assert comp.cells[0].f_pole_hz == 10e6


class TestHzToOmegaToQConversion:
    """#49: Explicit q = (2π*f)². Would fail under missing/duplicate 2π."""

    def test_conversion_factor(self):
        f = 10e6
        omega = _TWO_PI * f
        q = omega**2
        expected_q = (2 * math.pi * f) ** 2
        assert abs(q - expected_q) < 1.0  # within numerical precision
        # Check against wrong formula: q = f² (missing 2π)
        wrong_q = f**2
        assert abs(q - wrong_q) > 1e10  # very different


class TestNoQInPublicApi:
    """#50: Public functions accept Hz; q_m is internal only."""

    def test_no_q_params(self):
        import inspect

        for fn in [
            foster_reactance_hz,
            foster_derivative_hz,
            coefficients_to_components,
            compute_coefficient_bounds,
        ]:
            sig = inspect.signature(fn)
            param_names = set(sig.parameters.keys())
            assert "q_m" not in param_names, f"{fn.__name__} exposes q_m"
            assert "omega" not in param_names, f"{fn.__name__} exposes omega"


class TestFosterEvalHzApi:
    """#51: foster_reactance_hz accepts Hz, matches internal ω computation."""

    def test_hz_matches_omega(self):
        k0 = 1e9
        k_inf = 1e-7
        k_m = np.array([5e8])
        f_poles = np.array([8e6])
        f_eval = np.array([6e6])

        # Via Hz API
        x_hz = foster_reactance_hz(f_eval, k0, k_inf, k_m, f_poles)

        # Via internal ω function directly
        omega = _TWO_PI * f_eval
        q_m = (_TWO_PI * f_poles) ** 2
        x_omega = _foster_reactance_omega(omega, k0, k_inf, k_m, q_m)

        np.testing.assert_allclose(x_hz, x_omega, rtol=1e-14)


class TestDisabledEndpointNoneVsZero:
    """#51b: Disabled endpoints (None) and zero produce the same Foster value."""

    def test_none_equals_zero(self):
        k_m = np.array([1e9])
        f_poles = np.array([10e6])
        f_eval = np.array([6e6])

        # None (disabled)
        x_none = foster_reactance_hz(f_eval, None, None, k_m, f_poles)
        # Zero (explicit)
        x_zero = foster_reactance_hz(f_eval, 0.0, 0.0, k_m, f_poles)

        np.testing.assert_allclose(x_none, x_zero, atol=1e-14)


def test_prefix_recovers_pole_from_trivial_start():
    # [0, 0] -> [0, 0, 10] whole-prefix recomputation actually recovers the first required interval.
    # When [0, 0] is evaluated, it is deemed 'all-zero' and requires 0 poles.
    # When [0, 0, 10] is evaluated, it is no longer all-zero, and the 0 -> 0 transition
    # (where x_{i+1} <= x_i) now correctly identifies the required pole between target 0 and 1.
    f_targets = np.array([1e6, 2e6])
    x_targets_2 = np.array([0.0, 0.0])
    intervals_2 = find_required_pole_intervals(f_targets, x_targets_2)
    assert len(intervals_2) == 0

    f_targets_3 = np.array([1e6, 2e6, 3e6])
    x_targets_3 = np.array([0.0, 0.0, 10.0])
    intervals_3 = find_required_pole_intervals(f_targets_3, x_targets_3)
    assert len(intervals_3) == 1
    assert intervals_3[0].f_lo_hz == 1e6
    assert intervals_3[0].f_hi_hz == 2e6


def test_trivial_prefix_remains_trivial():
    # [0, 0, 0] remains trivial/no-pole.
    f_targets = np.array([1e6, 2e6, 3e6])
    x_targets = np.array([0.0, 0.0, 0.0])
    intervals = find_required_pole_intervals(f_targets, x_targets)
    assert len(intervals) == 0
