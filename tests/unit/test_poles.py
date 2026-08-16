"""Unit tests for pole validation and generation (Prompt 04A, tests #78-93)."""

from __future__ import annotations

import numpy as np
import pytest

from foster_eom.domain.topology import PoleMode
from foster_eom.foster.foster_form import RequiredPoleIntervalHz
from foster_eom.foster.poles import (
    PoleLayoutHz,
    PoleSpec,
    check_required_interval_feasibility,
    generate_pole_candidates,
    validate_poles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_F_TARGETS = np.array([5e6, 10e6, 15e6])


def _spec_fixed(poles: list[float], **kw) -> PoleSpec:
    return PoleSpec(
        mode=PoleMode.FIXED,
        fixed_poles_hz=tuple(poles),
        delta_f_target_min_hz=kw.get("excl", 100.0),
        delta_f_pole_min_hz=kw.get("sep", 100.0),
    )


def _spec_intervals(intervals: list[tuple[float, float]], **kw) -> PoleSpec:
    return PoleSpec(
        mode=PoleMode.INTERVALS,
        intervals_hz=tuple(intervals),
        delta_f_target_min_hz=kw.get("excl", 100.0),
        delta_f_pole_min_hz=kw.get("sep", 100.0),
    )


def _spec_auto(**kw) -> PoleSpec:
    return PoleSpec(
        mode=PoleMode.AUTO,
        allowed_band_hz=kw.get("band", (1e6, 30e6)),
        delta_f_target_min_hz=kw.get("excl", 100.0),
        delta_f_pole_min_hz=kw.get("sep", 100.0),
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestPoleOnTargetRejected:
    """#78: (C.5) Pole exactly at target -> rejected."""

    def test_on_target(self):
        layout = PoleLayoutHz(f_poles_hz=(10e6,))
        spec = _spec_fixed([10e6])
        val = validate_poles(layout, _F_TARGETS, spec)
        assert not val.valid
        assert any("too close" in v for v in val.violations)


class TestNearPoleExclusion:
    """#79: Within exclusion band -> rejected."""

    def test_near_target(self):
        layout = PoleLayoutHz(f_poles_hz=(10e6 + 50,))  # within 100 Hz excl
        spec = _spec_fixed([10e6 + 50], excl=100.0)
        val = validate_poles(layout, _F_TARGETS, spec)
        assert not val.valid


class TestFixedPolesNeverReordered:
    """#80: [10e6, 8e6] -> error, not sort."""

    def test_wrong_order(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            generate_pole_candidates(
                _spec_fixed([10e6, 8e6]),
                _F_TARGETS,
                n_cells=2,
            )


class TestFixedPolesValidatedExactly:
    """#81: Valid fixed -> unmodified."""

    def test_exact(self):
        candidates = generate_pole_candidates(
            _spec_fixed([7e6, 12e6]),
            _F_TARGETS,
            n_cells=2,
        )
        assert len(candidates) == 1
        assert candidates[0].f_poles_hz == (7e6, 12e6)


class TestIntervalDeterministicMidpoint:
    """#82: INTERVALS -> geometric midpoint (no RNG)."""

    def test_midpoint(self):
        import math

        spec = _spec_intervals([(6e6, 9e6), (11e6, 14e6)])
        candidates = generate_pole_candidates(spec, _F_TARGETS, n_cells=2)
        assert len(candidates) == 1
        fp = candidates[0].f_poles_hz
        assert abs(fp[0] - math.sqrt(6e6 * 9e6)) < 1.0
        assert abs(fp[1] - math.sqrt(11e6 * 14e6)) < 1.0


class TestIntervalOrderedByConstruction:
    """#83: Generated interval poles strictly ordered."""

    def test_ordered(self):
        spec = _spec_intervals([(6e6, 9e6), (11e6, 14e6), (16e6, 19e6)])
        candidates = generate_pole_candidates(spec, _F_TARGETS, n_cells=3)
        fp = candidates[0].f_poles_hz
        for i in range(len(fp) - 1):
            assert fp[i] < fp[i + 1]


class TestAutoRespectsRequiredIntervals:
    """#84: AUTO places poles in required intervals."""

    def test_auto(self):
        req = [RequiredPoleIntervalHz(7e6, 9e6)]
        spec = _spec_auto()
        candidates = generate_pole_candidates(spec, _F_TARGETS, n_cells=1, required_intervals=req)
        assert len(candidates) >= 1
        fp = candidates[0].f_poles_hz
        assert any(7e6 < f < 9e6 for f in fp)


class TestPoleSeparationEnforced:
    """#85: min_separation_hz respected."""

    def test_separation(self):
        layout = PoleLayoutHz(f_poles_hz=(7e6, 7.00005e6))
        spec = PoleSpec(
            mode=PoleMode.FIXED,
            fixed_poles_hz=(7e6, 7.00005e6),
            delta_f_target_min_hz=100.0,
            delta_f_pole_min_hz=100.0,
        )
        val = validate_poles(layout, np.array([5e6, 10e6]), spec)
        assert not val.valid
        assert any("separation" in v.lower() for v in val.violations)


class TestRequiredKGtMRejected:
    """#86: K > M -> infeasible."""

    def test_k_gt_m(self):
        req = [
            RequiredPoleIntervalHz(6e6, 9e6),
            RequiredPoleIntervalHz(11e6, 14e6),
            RequiredPoleIntervalHz(16e6, 19e6),
        ]
        result = check_required_interval_feasibility(
            req,
            n_cells=2,
            pole_spec=_spec_auto(),
            f_targets_hz=_F_TARGETS,
        )
        assert not result.feasible
        assert "3" in (result.reason or "")


class TestFixedPolesMissingRequiredInterval:
    """#87: Required interval has no fixed pole -> infeasible."""

    def test_missing(self):
        req = [RequiredPoleIntervalHz(6e6, 9e6)]  # required in (6-9)
        spec = _spec_fixed([12e6])  # pole at 12 MHz, not in (6-9)
        result = check_required_interval_feasibility(
            req,
            n_cells=1,
            pole_spec=spec,
            f_targets_hz=_F_TARGETS,
        )
        assert not result.feasible


class TestSurplusPoleDeterministic:
    """#88: M > K -> deterministic free-gap placement."""

    def test_surplus(self):
        req = [RequiredPoleIntervalHz(7e6, 9e6)]
        spec = _spec_auto()
        candidates = generate_pole_candidates(
            spec,
            _F_TARGETS,
            n_cells=3,
            required_intervals=req,
        )
        assert len(candidates) >= 1
        fp = candidates[0].f_poles_hz
        assert len(fp) == 3


class TestAllApisUseHz:
    """#89: No ω or q in pole API inputs/outputs."""

    def test_no_omega_q(self):
        import inspect

        for fn in [validate_poles, check_required_interval_feasibility, generate_pole_candidates]:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
            assert "omega" not in params
            assert "q_m" not in params


# ---------------------------------------------------------------------------
# Bipartite matching tests
# ---------------------------------------------------------------------------


class TestMatchingTwoCompeteOneSlotInfeasible:
    """#90: Two required intervals, one shared slot -> infeasible."""

    def test_competing(self):
        req = [
            RequiredPoleIntervalHz(7e6, 9e6),
            RequiredPoleIntervalHz(7.5e6, 8.5e6),  # overlaps first
        ]
        # Only one fixed pole at 8e6 -- both intervals want it
        spec = _spec_fixed([8e6])
        result = check_required_interval_feasibility(
            req,
            n_cells=1,
            pole_spec=spec,
            f_targets_hz=_F_TARGETS,
        )
        assert not result.feasible


class TestMatchingTwoToTwoFeasible:
    """#91: Two required, two distinct slots -> feasible."""

    def test_feasible(self):
        req = [
            RequiredPoleIntervalHz(6e6, 8e6),
            RequiredPoleIntervalHz(11e6, 14e6),
        ]
        spec = _spec_fixed([7e6, 12e6])
        result = check_required_interval_feasibility(
            req,
            n_cells=2,
            pole_spec=spec,
            f_targets_hz=_F_TARGETS,
        )
        assert result.feasible
        assert result.matching is not None
        assert len(result.matching) == 2


class TestMatchingGreedyFailsAugmentingSucceeds:
    """#92: Greedy left-to-right fails but valid matching exists.

    Setup:
    R1 = (6, 9)  -- can use slot A1=(6.5,7.5) or A2=(7,9)
    R2 = (7, 8)  -- can only use A2=(7,9)

    Greedy assigns R1->A1 (both work) then R2->A2 -> OK.
    But if greedy tried R1->A2 first, R2 could not use A1.
    The augmenting-path algorithm must find R1->A1, R2->A2.
    """

    def test_augmenting_path(self):
        req = [
            RequiredPoleIntervalHz(6e6, 9e6),  # R1: can use A1 or A2
            RequiredPoleIntervalHz(7e6, 8e6),  # R2: narrower, needs A2
        ]
        # Slot A1 = (6e6, 7.5e6), A2 = (7e6, 9e6)
        spec = _spec_intervals([(6e6, 7.5e6), (7e6, 9e6)], excl=0.0, sep=0.0)
        result = check_required_interval_feasibility(
            req,
            n_cells=2,
            pole_spec=spec,
            f_targets_hz=np.array([5e6, 10e6]),
        )
        assert result.feasible, f"Expected feasible, got: {result.reason}"


class TestMatchingReturnsExplicitPairs:
    """#93: Feasible matching returns (R_k, A_j) index pairs."""

    def test_pairs(self):
        req = [RequiredPoleIntervalHz(6e6, 9e6)]
        spec = _spec_fixed([7e6])
        result = check_required_interval_feasibility(
            req,
            n_cells=1,
            pole_spec=spec,
            f_targets_hz=_F_TARGETS,
        )
        assert result.feasible
        assert result.matching is not None
        assert len(result.matching) == 1
        rk, aj = result.matching[0]
        assert rk == 0
        assert aj == 0


class TestMatchingRequiresBacktracking:
    """#94: A perfect matching exists but fails placement; another succeeds."""

    def test_backtracking(self):
        # We want separation to kill the first matching but allow the second.
        from foster_eom.foster.poles import check_required_interval_feasibility

        req = [
            RequiredPoleIntervalHz(1e6, 4e6),
            RequiredPoleIntervalHz(2e6, 5e6),
        ]
        spec = _spec_intervals([(3e6, 7e6), (1e6, 5e6)], sep=2e6, excl=0.0)
        import numpy as np

        f_t = np.array([10e6])
        result = check_required_interval_feasibility(
            req, n_cells=2, pole_spec=spec, f_targets_hz=f_t
        )
        assert result.feasible
        assert len(result.matching) == 2
