"""Prompt 05 unit tests — domain.py (ContinuousOptimizationDomain).

Tests cover: k-box bounds, FIXED-FIXED separation, domain hash determinism.
Full integration tests are in test_engine.py.
"""

from __future__ import annotations

import pytest

from foster_eom.domain.component import ContinuousLimits
from foster_eom.optimize.domain import (
    _check_fixed_fixed_separation,
    _compute_k_box_bounds,
    _domain_hash,
)

# ---------------------------------------------------------------------------
# Tests for _compute_k_box_bounds
# ---------------------------------------------------------------------------


def _limits(
    c_min=1e-12, c_max=1e-6, l_min=1e-9, l_max=1e-3
) -> ContinuousLimits:
    return ContinuousLimits(
        c_min_f=c_min, c_max_f=c_max,
        l_min_h=l_min, l_max_h=l_max,
    )


class TestKBoxBounds:
    def test_single_cell_valid(self) -> None:
        lims = _limits()
        pole_regions = ((1e6, 10e6),)
        bounds, feasible, reason = _compute_k_box_bounds(1, pole_regions, lims)
        assert feasible
        assert reason is None
        assert len(bounds) == 1
        lo, hi = bounds[0]
        assert lo > 0
        assert hi > lo

    def test_infeasible_when_kmin_gt_kmax(self) -> None:
        """Very small component limits should make k-box infeasible for high f_pole."""
        lims = ContinuousLimits(c_min_f=1e-3, c_max_f=1e-2, l_min_h=1e-3, l_max_h=1e-2)
        # At f=10 GHz, q_lo = (2π*1e10)^2 ≈ 4e21; k_box_min = q_lo*l_min ≫ 1/c_min
        pole_regions = ((10e9, 10e9),)
        bounds, feasible, reason = _compute_k_box_bounds(1, pole_regions, lims)
        assert not feasible
        assert reason is not None

    def test_zero_pole_region_uses_c_bound_only(self) -> None:
        """f_lo = 0 → q_lo = 0 → k_box_min = 1/c_max."""
        lims = _limits()
        pole_regions = ((0.0, 0.0),)
        bounds, feasible, reason = _compute_k_box_bounds(1, pole_regions, lims)
        lo, hi = bounds[0]
        assert lo == pytest.approx(1.0 / lims.c_max_f, rel=1e-9)


# ---------------------------------------------------------------------------
# Tests for _check_fixed_fixed_separation
# ---------------------------------------------------------------------------


class TestFixedFixedSeparation:
    def test_no_adjacent_fixed_passes(self) -> None:
        # Both wide intervals — not FIXED-FIXED
        regions = ((1e6, 10e6), (15e6, 20e6))
        ok, reason = _check_fixed_fixed_separation(
            f_poles=(5e6, 18e6),
            pole_regions=regions,
            delta_f_min=100e3,
        )
        assert ok

    def test_fixed_fixed_adequate_separation_passes(self) -> None:
        regions = ((5e6, 5e6), (6e6, 6e6))  # both point
        ok, reason = _check_fixed_fixed_separation(
            f_poles=(5e6, 6e6),
            pole_regions=regions,
            delta_f_min=200e3,  # 1 MHz > 200 kHz
        )
        assert ok

    def test_fixed_fixed_too_close_fails(self) -> None:
        regions = ((5e6, 5e6), (5.1e6, 5.1e6))  # 100 kHz apart
        ok, reason = _check_fixed_fixed_separation(
            f_poles=(5e6, 5.1e6),
            pole_regions=regions,
            delta_f_min=200e3,  # need 200 kHz
        )
        assert not ok
        assert reason is not None


# ---------------------------------------------------------------------------
# Tests for domain hash determinism
# ---------------------------------------------------------------------------


class TestDomainHash:
    def _make_topo(self, cells1: int = 1, cells2: int = 1) -> TopologyCandidate:
        from foster_eom.domain.topology import LOrientation
        from foster_eom.foster.topology_enum import TopologyCandidate
        return TopologyCandidate(
            orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
            branch1_cells=cells1, branch2_cells=cells2,
            branch1_has_c0=False, branch1_has_linf=False,
            branch2_has_c0=False, branch2_has_linf=False,
            branch1_n_coefficients=cells1,
            branch2_n_coefficients=cells2,
            n_reactive=2 * cells1 + 2 * cells2,
            structurally_valid=True,
            prune_reason=None,
        )

    def test_same_input_same_hash(self) -> None:
        from foster_eom.foster.schmidt import BranchRealization
        topo = self._make_topo(1, 1)
        h1 = _domain_hash(
            orientation_value="schmidt_shunt_then_series",
            topology=topo,
            branch1_realization=BranchRealization.FINITE_FOSTER,
            branch2_realization=BranchRealization.FINITE_FOSTER,
            pole_regions_branch1=((1e6, 10e6),),
            pole_regions_branch2=((5e6, 50e6),),
            n_movable_b1=1,
            n_movable_b2=1,
        )
        h2 = _domain_hash(
            orientation_value="schmidt_shunt_then_series",
            topology=topo,
            branch1_realization=BranchRealization.FINITE_FOSTER,
            branch2_realization=BranchRealization.FINITE_FOSTER,
            pole_regions_branch1=((1e6, 10e6),),
            pole_regions_branch2=((5e6, 50e6),),
            n_movable_b1=1,
            n_movable_b2=1,
        )
        assert h1 == h2

    def test_different_topology_different_hash(self) -> None:
        from foster_eom.foster.schmidt import BranchRealization
        topo1 = self._make_topo(1, 1)
        topo2 = self._make_topo(2, 1)
        h1 = _domain_hash(
            "schmidt_shunt_then_series", topo1,
            BranchRealization.FINITE_FOSTER, BranchRealization.FINITE_FOSTER,
            ((1e6, 10e6),), ((5e6, 50e6),), 1, 1,
        )
        h2 = _domain_hash(
            "schmidt_shunt_then_series", topo2,
            BranchRealization.FINITE_FOSTER, BranchRealization.FINITE_FOSTER,
            ((1e6, 10e6), (20e6, 30e6)), ((5e6, 50e6),), 2, 1,
        )
        assert h1 != h2
