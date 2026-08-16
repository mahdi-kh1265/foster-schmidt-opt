"""Tests for sign-pattern enumeration (Prompt 04B)."""

from __future__ import annotations

import numpy as np
import pytest

from foster_eom.domain.topology import LOrientation
from foster_eom.foster.poles import PoleMode as InternalPoleMode
from foster_eom.foster.poles import PoleSpec as InternalPoleSpec
from foster_eom.foster.schmidt import (
    BranchRealization,
    schmidt_standard_targets,
)
from foster_eom.foster.sign_search import (
    SignSearchConstraints,
    enumerate_sign_patterns,
)


def _default_constraints(
    max_cells: int = 3,
) -> SignSearchConstraints:
    ps = InternalPoleSpec(mode=InternalPoleMode.AUTO)
    return SignSearchConstraints(
        branch1_min_cells=0,
        branch1_max_cells=max_cells,
        branch2_min_cells=0,
        branch2_max_cells=max_cells,
        pole_spec_branch1=ps,
        pole_spec_branch2=ps,
    )


class TestSignPatternExhaustive:
    """Exhaustive search for N <= 8."""

    def test_single_target_two_patterns(self) -> None:
        """One target → 2 sign combos, both should be evaluated."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([25.0 + 30j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        assert schmidt.all_valid

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        assert result.diagnostics.search_exhaustive
        assert result.diagnostics.n_total_evaluated == 2

    def test_three_targets_exhaustive(self) -> None:
        """Three targets → 2^3 = 8 sign combos evaluated exhaustively."""
        r_match = 50.0
        f_hz = np.array([9e6, 10e6, 11e6])
        z_load = np.array(
            [
                25.0 + 30j,
                30.0 + 10j,
                25.0 - 20j,
            ]
        )
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible for this test case")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        assert result.diagnostics.search_exhaustive
        assert result.diagnostics.n_total_evaluated == 8

    def test_infeasible_schmidt_returns_empty(self) -> None:
        """All-infeasible Schmidt → no sign patterns."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([-50.0 + 0j])  # Negative real → infeasible
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        # This should be infeasible
        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        # Empty result
        assert len(result.patterns) == 0

    def test_all_patterns_carry_correct_orientation(self) -> None:
        """All returned sign patterns have the correct orientation."""
        r_match = 50.0
        f_hz = np.array([10e6, 11e6])
        z_load = np.array([25.0 + 10j, 30.0 + 5j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        for p_info in result.patterns:
            assert p_info.pattern.orientation == LOrientation.SCHMIDT_SHUNT_THEN_SERIES


class TestSignPatternDiagnostics:
    """Diagnostics reporting."""

    def test_prune_counts_sum_correctly(self) -> None:
        """structural_prune_counts values must sum to n_pruned_structural."""
        r_match = 50.0
        f_hz = np.array([9e6, 10e6, 11e6])
        z_load = np.array([25.0 + 30j, 30.0 + 10j, 25.0 - 20j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        d = result.diagnostics
        assert sum(d.structural_prune_counts.values()) == d.n_pruned_structural

    def test_evaluated_equals_pruned_plus_accepted(self) -> None:
        """n_total_evaluated == n_pruned + len(patterns)."""
        r_match = 50.0
        f_hz = np.array([10e6, 11e6])
        z_load = np.array([25.0 + 10j, 30.0 + 5j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        d = result.diagnostics
        assert d.n_total_evaluated == d.n_pruned_structural + len(result.patterns)


class TestSignPatternBranchRealization:
    """Branch realization classification in sign patterns."""

    def test_finite_foster_branches_have_targets(self) -> None:
        """FINITE_FOSTER branches have non-empty target tuples."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([25.0 + 10j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(),
        )
        for p_info in result.patterns:
            pat = p_info.pattern
            if pat.branch1_realization == BranchRealization.FINITE_FOSTER:
                assert len(pat.shunt_targets) > 0
            if pat.branch2_realization == BranchRealization.FINITE_FOSTER:
                assert len(pat.series_targets) > 0
