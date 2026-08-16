"""Unit tests for Schmidt target-reactance solver (Prompt 04A, tests #1-37)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from foster_eom.foster.schmidt import (
    BranchRealization,
    FosterBranchTolerances,
    ReactanceTarget,
    ReactanceTargetState,
    SchmidtTolerances,
    TargetFeasibility,
    classify_branch_realization,
    schmidt_dual_targets,
    schmidt_standard_targets,
    validate_branch_realization_legality,
    validate_schmidt_targets_algebraic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RM = 50.0  # R_match default


def _zl(rl: float, xl: float = 0.0) -> np.ndarray:
    return np.array([rl + 1j * xl])


def _f1() -> np.ndarray:
    return np.array([10e6])


# ---------------------------------------------------------------------------
# Standard orientation
# ---------------------------------------------------------------------------


class TestStandardHandComputed:
    """#1: Hand-computed complex load."""

    def test_standard_hand_computed(self):
        rl, xl = 20.0, 15.0
        result = schmidt_standard_targets(_RM, _zl(rl, xl), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.ORDINARY
        # rho = 20/50 = 0.4
        assert abs(pt.rho_or_g - 0.4) < 1e-14
        # X_2 = +/- 50 * sqrt(0.4/0.6) ~= +/-40.8248...
        expected_x2 = _RM * math.sqrt(0.4 / 0.6)
        assert abs(pt.x_shunt_plus.value_ohm - expected_x2) < 1e-8  # type: ignore[arg-type]
        assert abs(pt.x_shunt_minus.value_ohm + expected_x2) < 1e-8  # type: ignore[arg-type]


class TestStandardBothSignsValid:
    """#2: Both signs produce Z_in = R_match."""

    def test_both_signs(self):
        result = schmidt_standard_targets(_RM, _zl(30.0, 10.0), _f1())
        ok_plus, _ = validate_schmidt_targets_algebraic(result, (1,))
        ok_minus, _ = validate_schmidt_targets_algebraic(result, (-1,))
        assert ok_plus
        assert ok_minus


class TestStandardAlgebraicMultiFreq:
    """#3: Multiple frequencies, algebraic verification."""

    def test_multi_freq(self):
        f = np.array([7.3e6, 12.1e6, 15.8e6])
        zl = np.array([25 + 5j, 35 - 10j, 10 + 20j])
        result = schmidt_standard_targets(_RM, zl, f)
        assert result.all_valid
        ok, zin = validate_schmidt_targets_algebraic(result, (1, 1, 1))
        assert ok
        for z in zin:
            assert abs(z - _RM) / _RM < 1e-8


class TestStandardDegenerateExact:
    """#4: rho = 1.0 exactly -> OPEN, X_1 = -X_L."""

    def test_exact(self):
        result = schmidt_standard_targets(_RM, _zl(50.0, 20.0), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.DEGENERATE
        assert pt.x_shunt_plus.state == ReactanceTargetState.OPEN_CIRCUIT
        assert pt.x_series_for_plus.value_ohm == pytest.approx(-20.0)


class TestStandardDegenerateWithinTolerance:
    """#5: rho = 1 - eps/2 -> degenerate."""

    def test_within_tol(self):
        tol = SchmidtTolerances(epsilon_rho=1e-10)
        rl = _RM * (1 - tol.epsilon_rho / 2)
        result = schmidt_standard_targets(_RM, _zl(rl), _f1(), tolerances=tol)
        assert result.points[0].feasibility == TargetFeasibility.DEGENERATE


class TestStandardOrdinaryJustOutsideTolerance:
    """#6: rho = 1 - 2eps -> ordinary."""

    def test_just_outside(self):
        tol = SchmidtTolerances(epsilon_rho=1e-10)
        rl = _RM * (1 - 2 * tol.epsilon_rho)
        result = schmidt_standard_targets(_RM, _zl(rl), _f1(), tolerances=tol)
        assert result.points[0].feasibility == TargetFeasibility.ORDINARY


class TestStandardDegenerateAlgebraicCompatibleTol:
    """#7: Tolerance-snapped degenerate passes algebraic check."""

    def test_compatible_tol(self):
        tol = SchmidtTolerances(epsilon_rho=1e-6)
        rl = _RM * (1 - tol.epsilon_rho / 2)
        result = schmidt_standard_targets(_RM, _zl(rl, 30.0), _f1(), tolerances=tol)
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.DEGENERATE
        # Use compatible tolerance (larger than epsilon_rho)
        ok, _ = validate_schmidt_targets_algebraic(result, (1,), tol=1e-5)
        assert ok


class TestStandardDegenerateXlNonzero:
    """#8: X_L != 0: X_1 = -X_L."""

    def test_xl_cancellation(self):
        result = schmidt_standard_targets(_RM, _zl(50.0, -42.7), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.DEGENERATE
        assert pt.x_series_for_plus.value_ohm == pytest.approx(42.7)


class TestStandardInfeasibleRhoGt1PlusTol:
    """#9: rho > 1 + eps -> infeasible."""

    def test_rho_too_large(self):
        result = schmidt_standard_targets(_RM, _zl(60.0), _f1())
        assert result.points[0].feasibility == TargetFeasibility.INFEASIBLE
        assert not result.all_valid


class TestStandardDegenerateRhoJustAbove1:
    """#10: rho = 1 + eps/2 -> degenerate (not infeasible)."""

    def test_just_above(self):
        tol = SchmidtTolerances(epsilon_rho=1e-10)
        rl = _RM * (1 + tol.epsilon_rho / 2)
        result = schmidt_standard_targets(_RM, _zl(rl), _f1(), tolerances=tol)
        assert result.points[0].feasibility == TargetFeasibility.DEGENERATE


class TestStandardInfeasibleRlZero:
    """#11: R_L = 0 -> infeasible."""

    def test_zero(self):
        result = schmidt_standard_targets(_RM, _zl(0.0, 10.0), _f1())
        assert result.points[0].feasibility == TargetFeasibility.INFEASIBLE


class TestStandardInfeasibleRlNegative:
    """#12: R_L < 0 -> infeasible."""

    def test_negative(self):
        result = schmidt_standard_targets(_RM, _zl(-5.0), _f1())
        assert result.points[0].feasibility == TargetFeasibility.INFEASIBLE


class TestStandardNearBoundaryRadicandClamped:
    """#13: Ordinary region, no negative radicand surprise.

    rho well below degenerate boundary -> radicand must be genuinely positive.
    A truly ordinary point should never produce a negative radicand.
    """

    def test_ordinary_radicand_positive(self):
        # rho = 0.5 -- solidly ordinary
        result = schmidt_standard_targets(_RM, _zl(25.0), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.ORDINARY
        rho = pt.rho_or_g
        assert rho * (1 - rho) > 0


# ---------------------------------------------------------------------------
# Dual orientation
# ---------------------------------------------------------------------------


class TestDualHandComputed:
    """#14: Complex load with g < 1."""

    def test_dual_hand(self):
        zl = 100.0 + 50j  # G = 100/(100²+50²) = 0.008, g = 0.4
        result = schmidt_dual_targets(_RM, np.array([zl]), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.ORDINARY
        g = pt.rho_or_g
        # G = 100 / (100² + 50²) = 100/12500 = 0.008
        # g = 0.008 * 50 = 0.4
        assert abs(g - 0.4) < 1e-12


class TestDualBothSignsValid:
    """#15: Both signs -> Z_in = R_match."""

    def test_both(self):
        zl = np.array([100.0 + 30j])
        result = schmidt_dual_targets(_RM, zl, _f1())
        ok_p, _ = validate_schmidt_targets_algebraic(result, (1,))
        ok_m, _ = validate_schmidt_targets_algebraic(result, (-1,))
        assert ok_p
        assert ok_m


class TestDualAlgebraicMultiFreq:
    """#16: Multiple frequencies, dual."""

    def test_multi(self):
        f = np.array([7.3e6, 12.1e6, 15.8e6])
        # Use loads with G < 1/R_match for all
        zl = np.array([200 + 100j, 150 - 50j, 300 + 200j])
        result = schmidt_dual_targets(_RM, zl, f)
        assert result.all_valid
        ok, _ = validate_schmidt_targets_algebraic(result, (1, 1, 1))
        assert ok


class TestDualDegenerateExact:
    """#17: g = 1.0 exactly -> X_s = 0."""

    def test_exact(self):
        # Need G = 1/R_match = 0.02.  Z_L = 1/(0.02 + jB).
        # For B=0: Z_L = 50.  Then G = 1/50 = 0.02, g = 1.
        result = schmidt_dual_targets(_RM, _zl(50.0, 0.0), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.DEGENERATE
        assert pt.x_series_for_plus.value_ohm == pytest.approx(0.0, abs=1e-12)


class TestDualDegenerateWithinTolerance:
    """#18: g ~= 1 (within tol) -> degenerate."""

    def test_within_tol(self):
        tol = SchmidtTolerances(epsilon_g=1e-10)
        # Need g = 1 - eps/2 -> G = (1 - eps/2)/R_match
        g_target = 1.0 - tol.epsilon_g / 2
        g_load = g_target / _RM
        # Z_L = 1/G (purely real)
        zl = np.array([1.0 / g_load])
        result = schmidt_dual_targets(_RM, zl, _f1(), tolerances=tol)
        assert result.points[0].feasibility == TargetFeasibility.DEGENERATE


class TestDualOrdinaryJustOutsideTolerance:
    """#19: g = 1 - 2eps -> ordinary."""

    def test_just_outside(self):
        tol = SchmidtTolerances(epsilon_g=1e-10)
        g_target = 1.0 - 2 * tol.epsilon_g
        g_load = g_target / _RM
        zl = np.array([1.0 / g_load])
        result = schmidt_dual_targets(_RM, zl, _f1(), tolerances=tol)
        assert result.points[0].feasibility == TargetFeasibility.ORDINARY


class TestDualDegenerateAlgebraicCompatibleTol:
    """#20: Tolerance-snapped dual degenerate passes check."""

    def test_compatible_tol(self):
        tol = SchmidtTolerances(epsilon_g=1e-6)
        # Need g = G*R_match = 1 - eps/2.  For a purely real load,
        # G = 1/R_L, so R_L = R_match / g_target.
        g_target = 1.0 - tol.epsilon_g / 2
        rl = _RM / g_target  # ~= 50 Ω
        zl = np.array([rl + 10j])
        # With complex load: G = rl/(rl² + xl²).  Need to construct
        # the load so that G*R_match ~= 1 within tolerance.
        # G = rl/(rl² + 100), g = G*50
        # For rl ~= 50, g = 50*50/(50²+100) = 2500/2600 ~= 0.9615
        # That's far from 1.  Use purely real load instead.
        zl = np.array([rl + 0j])
        result = schmidt_dual_targets(_RM, zl, _f1(), tolerances=tol)
        assert result.points[0].feasibility == TargetFeasibility.DEGENERATE
        ok, _ = validate_schmidt_targets_algebraic(result, (1,), tol=1e-5)
        assert ok


class TestDualDegenerateBpZero:
    """#21: B_p ~= 0 -> X_p = OPEN."""

    def test_bp_zero(self):
        # G = 1/R_match, B = 0 -> B_p = B_t - B = 0 - 0 = 0 -> OPEN
        result = schmidt_dual_targets(_RM, _zl(50.0, 0.0), _f1())
        pt = result.points[0]
        assert pt.feasibility == TargetFeasibility.DEGENERATE
        # Both shunt targets should be OPEN (B=0 and degenerate)
        assert pt.x_shunt_plus.state == ReactanceTargetState.OPEN_CIRCUIT


class TestDualInfeasibleGGt1PlusTol:
    """#22: g > 1 + eps -> infeasible."""

    def test_too_large(self):
        # G > 1/R_match -> g > 1.  Z_L = 10+0j -> G = 0.1, g = 5 >> 1
        result = schmidt_dual_targets(_RM, _zl(10.0, 0.0), _f1())
        assert result.points[0].feasibility == TargetFeasibility.INFEASIBLE


class TestDualComplexLoadRlLtRmatchFeasible:
    """#23: R_L < R_match but large X_L -> dual feasible."""

    def test_complex_feasible(self):
        # Z_L = 30 + 200j -> G = 30/(30²+200²) = 30/40900 ~= 7.33e-4
        # g = 0.000733 * 50 ~= 0.0367 < 1 -> ordinary
        zl = np.array([30.0 + 200j])
        result = schmidt_dual_targets(_RM, zl, _f1())
        assert result.points[0].feasibility == TargetFeasibility.ORDINARY
        assert result.all_valid


class TestDualNearBoundaryRadicandClamped:
    """#24: Dual ordinary region, radicand positive."""

    def test_positive_radicand(self):
        zl = np.array([200.0 + 0j])  # G = 0.005, g = 0.25
        result = schmidt_dual_targets(_RM, zl, _f1())
        pt = result.points[0]
        g = pt.rho_or_g
        assert g * (1 - g) > 0


class TestRmatchNot50:
    """#25: R_match = 75Ω."""

    def test_75_ohm(self):
        rm = 75.0
        result = schmidt_standard_targets(rm, _zl(40.0, 10.0), _f1())
        ok, _ = validate_schmidt_targets_algebraic(result, (1,))
        assert ok

        # Dual: need G < 1/75
        zl_dual = np.array([200.0 + 50j])
        result_d = schmidt_dual_targets(rm, zl_dual, _f1())
        ok_d, _ = validate_schmidt_targets_algebraic(result_d, (1,))
        assert ok_d


class TestArbitraryNonHarmonicFrequencies:
    """#26: 7.3, 12.1, 15.8 MHz."""

    def test_non_harmonic(self):
        f = np.array([7.3e6, 12.1e6, 15.8e6])
        zl = np.array([30 + 10j, 25 - 5j, 40 + 15j])
        result = schmidt_standard_targets(_RM, zl, f)
        assert result.all_valid
        ok, _ = validate_schmidt_targets_algebraic(result, (1, -1, 1))
        assert ok


# ---------------------------------------------------------------------------
# ReactanceTarget invariant validation
# ---------------------------------------------------------------------------


class TestReactanceTargetFiniteRequiresFloat:
    """#27: FINITE + None -> ValueError."""

    def test_finite_none(self):
        with pytest.raises(ValueError, match="FINITE"):
            ReactanceTarget(10e6, None, ReactanceTargetState.FINITE)


class TestReactanceTargetOpenRequiresNone:
    """#28: OPEN + float -> ValueError."""

    def test_open_float(self):
        with pytest.raises(ValueError, match="OPEN_CIRCUIT"):
            ReactanceTarget(10e6, 42.0, ReactanceTargetState.OPEN_CIRCUIT)


class TestReactanceTargetFiniteInfRaises:
    """#29: FINITE + inf -> ValueError."""

    def test_finite_inf(self):
        with pytest.raises(ValueError, match="FINITE"):
            ReactanceTarget(10e6, float("inf"), ReactanceTargetState.FINITE)


# ---------------------------------------------------------------------------
# Branch realization classification
# ---------------------------------------------------------------------------


class TestBranchAllOpenOmitted:
    """#30: All OPEN -> OPEN_OMITTED."""

    def test_all_open(self):
        targets = (
            ReactanceTarget(10e6, None, ReactanceTargetState.OPEN_CIRCUIT),
            ReactanceTarget(11e6, None, ReactanceTargetState.OPEN_CIRCUIT),
        )
        assert (
            classify_branch_realization(targets, _RM, is_series=True)
            == BranchRealization.OPEN_OMITTED
        )


class TestBranchAllZeroWire:
    """#31: All zero (within tol) -> ZERO_IMPEDANCE."""

    def test_all_zero(self):
        targets = (
            ReactanceTarget(10e6, 0.0, ReactanceTargetState.FINITE),
            ReactanceTarget(11e6, 0.001, ReactanceTargetState.FINITE),
        )
        assert (
            classify_branch_realization(targets, _RM, is_series=True)
            == BranchRealization.ZERO_IMPEDANCE
        )


class TestBranchMixedOpenFiniteInfeasible:
    """#32: Mixed OPEN + FINITE -> ValueError."""

    def test_mixed(self):
        targets = (
            ReactanceTarget(10e6, 42.0, ReactanceTargetState.FINITE),
            ReactanceTarget(11e6, None, ReactanceTargetState.OPEN_CIRCUIT),
        )
        with pytest.raises(ValueError, match="Mixed OPEN"):
            classify_branch_realization(targets, _RM, is_series=True)


class TestZeroImpedanceShuntRejected:
    """#33: ZERO_IMPEDANCE shunt -> rejected."""

    def test_shunt_short(self):
        legal, reason = validate_branch_realization_legality(
            BranchRealization.ZERO_IMPEDANCE, is_series=False
        )
        assert not legal
        assert "short" in reason.lower()  # type: ignore[union-attr]


class TestOpenOmittedSeriesRejected:
    """#34: OPEN_OMITTED series -> rejected."""

    def test_series_disconnect(self):
        legal, reason = validate_branch_realization_legality(
            BranchRealization.OPEN_OMITTED, is_series=True
        )
        assert not legal
        assert "disconnect" in reason.lower()  # type: ignore[union-attr]


class TestDimensionlessClassificationGapFree:
    """#35: Sweep rho from 0 to 2 -- every value classified, no gap."""

    def test_gap_free(self):
        rhos = np.linspace(0.01, 2.0, 1000)
        tol = SchmidtTolerances(epsilon_rho=1e-4)
        for rho in rhos:
            rl = rho * _RM
            result = schmidt_standard_targets(_RM, _zl(rl), _f1(), tolerances=tol)
            pt = result.points[0]
            assert pt.feasibility in (
                TargetFeasibility.ORDINARY,
                TargetFeasibility.DEGENERATE,
                TargetFeasibility.INFEASIBLE,
            )


class TestZeroTargetToleranceJustInside:
    """#36: max|X| just inside zero threshold -> ZERO_IMPEDANCE."""

    def test_just_inside(self):
        bt = FosterBranchTolerances(x_zero_abs=0.01, x_zero_rel=1e-6)
        threshold = bt.x_zero_abs + bt.x_zero_rel * _RM
        val = threshold - 1e-10  # just inside
        targets = (
            ReactanceTarget(10e6, val, ReactanceTargetState.FINITE),
            ReactanceTarget(11e6, -val, ReactanceTargetState.FINITE),
        )
        assert (
            classify_branch_realization(targets, _RM, is_series=True, branch_tol=bt)
            == BranchRealization.ZERO_IMPEDANCE
        )


class TestZeroTargetToleranceJustOutside:
    """#37: max|X| just outside zero threshold -> FINITE_FOSTER."""

    def test_just_outside(self):
        bt = FosterBranchTolerances(x_zero_abs=0.01, x_zero_rel=1e-6)
        threshold = bt.x_zero_abs + bt.x_zero_rel * _RM
        val = threshold + 0.1  # outside
        targets = (
            ReactanceTarget(10e6, val, ReactanceTargetState.FINITE),
            ReactanceTarget(11e6, 0.0, ReactanceTargetState.FINITE),
        )
        assert (
            classify_branch_realization(targets, _RM, is_series=True, branch_tol=bt)
            == BranchRealization.FINITE_FOSTER
        )


class TestShuntSmallFiniteFoster:
    """#36: A near-zero finite shunt target remains FINITE_FOSTER (not ZERO_IMPEDANCE)."""

    def test_small_finite_shunt(self):
        targets = (
            ReactanceTarget(10e6, 1e-10, ReactanceTargetState.FINITE),
            ReactanceTarget(11e6, -1e-10, ReactanceTargetState.FINITE),
        )
        # For series, it would be ZERO_IMPEDANCE
        assert (
            classify_branch_realization(targets, 50.0, is_series=True)
            == BranchRealization.ZERO_IMPEDANCE
        )
        # For shunt, it must remain FINITE_FOSTER to avoid false invalidation
        assert (
            classify_branch_realization(targets, 50.0, is_series=False)
            == BranchRealization.FINITE_FOSTER
        )


class TestShuntExactZeroInvalid:
    """#37: An exact zero shunt target becomes ZERO_IMPEDANCE (structurally invalid)."""

    def test_exact_zero_shunt(self):
        targets = (
            ReactanceTarget(10e6, 0.0, ReactanceTargetState.FINITE),
            ReactanceTarget(11e6, 0.0, ReactanceTargetState.FINITE),
        )
        # Even for shunt, exact 0.0 means ZERO_IMPEDANCE
        realization = classify_branch_realization(targets, 50.0, is_series=False)
        assert realization == BranchRealization.ZERO_IMPEDANCE

        # And validate_branch_realization_legality rejects it
        from foster_eom.foster.schmidt import validate_branch_realization_legality

        legal, _ = validate_branch_realization_legality(realization, is_series=False)
        assert not legal
