"""Tests for topology enumeration (Prompt 04B)."""

from __future__ import annotations

import numpy as np
import pytest

from foster_eom.domain.topology import LOrientation, TopologySearchSpec
from foster_eom.foster.poles import PoleMode as InternalPoleMode
from foster_eom.foster.poles import PoleSpec as InternalPoleSpec
from foster_eom.foster.schmidt import BranchRealization, schmidt_standard_targets
from foster_eom.foster.sign_search import (
    SignSearchConstraints,
    enumerate_sign_patterns,
)
from foster_eom.foster.topology_enum import enumerate_topologies


def _default_topo_spec(**kw) -> TopologySearchSpec:
    return TopologySearchSpec(**kw)


def _default_constraints(max_cells: int = 3) -> SignSearchConstraints:
    ps = InternalPoleSpec(mode=InternalPoleMode.AUTO)
    return SignSearchConstraints(
        branch1_min_cells=0,
        branch1_max_cells=max_cells,
        branch2_min_cells=0,
        branch2_max_cells=max_cells,
        pole_spec_branch1=ps,
        pole_spec_branch2=ps,
    )


class TestTopologyEnumeration:
    """Basic topology enumeration tests."""

    def test_orientation_mismatch_raises(self) -> None:
        """Topology spec orientation must match sign pattern orientation."""
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
        if not result.patterns:
            pytest.skip("No patterns found")

        sign_info = result.patterns[0]
        # Create topo_spec with different orientation
        topo_spec = TopologySearchSpec(
            orientations=[LOrientation.ALTERNATE_L_ORIENTATION],
        )
        with pytest.raises(ValueError, match="Orientation mismatch"):
            enumerate_topologies(topo_spec, sign_info)

    def test_finite_foster_generates_candidates(self) -> None:
        """FINITE_FOSTER branches should generate topology candidates."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([25.0 + 10j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(max_cells=2),
        )
        if not result.patterns:
            pytest.skip("No patterns found")

        topo_spec = _default_topo_spec(
            branch1_cells_min=1,
            branch1_cells_max=2,
            branch2_cells_min=1,
            branch2_cells_max=2,
        )
        sign_info = result.patterns[0]
        candidates = enumerate_topologies(topo_spec, sign_info)
        # At least one candidate should exist
        assert len(candidates) > 0

    def test_all_candidates_structurally_valid(self) -> None:
        """All returned candidates must be structurally valid."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([25.0 + 10j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(max_cells=2),
        )
        if not result.patterns:
            pytest.skip("No patterns found")

        topo_spec = _default_topo_spec(
            branch1_cells_min=1,
            branch1_cells_max=2,
            branch2_cells_min=1,
            branch2_cells_max=2,
        )
        for sign_info in result.patterns:
            candidates = enumerate_topologies(topo_spec, sign_info)
            for c in candidates:
                assert c.structurally_valid
                assert c.prune_reason is None

    def test_max_reactive_pruning(self) -> None:
        """Candidates exceeding max_total_reactive_components are pruned."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([25.0 + 10j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(max_cells=2),
        )
        if not result.patterns:
            pytest.skip("No patterns found")

        # Very tight reactive cap
        topo_spec = _default_topo_spec(
            branch1_cells_min=1,
            branch1_cells_max=2,
            branch2_cells_min=1,
            branch2_cells_max=2,
            max_total_reactive_components=2,  # Very tight
        )
        sign_info = result.patterns[0]
        candidates = enumerate_topologies(topo_spec, sign_info)
        for c in candidates:
            assert c.n_reactive <= 2

    def test_trivial_branches_have_zero_cells(self) -> None:
        """Non-FINITE_FOSTER branches generate candidates with 0 cells."""
        r_match = 50.0
        f_hz = np.array([10e6])
        z_load = np.array([25.0 + 10j])
        schmidt = schmidt_standard_targets(r_match, z_load, f_hz)
        if not schmidt.all_valid:
            pytest.skip("Schmidt infeasible")

        result = enumerate_sign_patterns(
            schmidt,
            _default_constraints(max_cells=2),
        )
        for sign_info in result.patterns:
            topo_spec = _default_topo_spec(
                branch1_cells_min=1,
                branch1_cells_max=2,
                branch2_cells_min=1,
                branch2_cells_max=2,
            )
            candidates = enumerate_topologies(topo_spec, sign_info)
            for c in candidates:
                pat = sign_info.pattern
                if pat.branch1_realization != BranchRealization.FINITE_FOSTER:
                    assert c.branch1_cells == 0
                if pat.branch2_realization != BranchRealization.FINITE_FOSTER:
                    assert c.branch2_cells == 0
