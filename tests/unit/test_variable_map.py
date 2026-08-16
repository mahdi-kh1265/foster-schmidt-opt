"""Prompt 05 unit tests — variable_map.py."""

from __future__ import annotations

import math
import numpy as np
import pytest

from foster_eom.optimize.variable_map import (
    DecisionVariableMapper,
    VariableDescriptor,
    build_variable_mapper,
    BranchCoordinates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi


def _simple_mapper_1cell() -> DecisionVariableMapper:
    """One cell per branch, movable pole, no endpoints."""
    return build_variable_mapper(
        branch1_n_cells=1,
        branch1_has_c0=False,
        branch1_has_linf=False,
        branch1_pole_regions=((1e6, 10e6),),
        branch1_k_box_bounds=((1.0, 1e4),),
        branch1_k0_bounds=None,
        branch1_kinf_bounds=None,
        branch1_fixed_k0=None,
        branch1_fixed_kinf=None,
        branch1_fixed_k_residues=(None,),
        branch1_fixed_f_poles_hz=(None,),
        branch2_n_cells=1,
        branch2_has_c0=False,
        branch2_has_linf=False,
        branch2_pole_regions=((5e6, 50e6),),
        branch2_k_box_bounds=((1.0, 1e4),),
        branch2_k0_bounds=None,
        branch2_kinf_bounds=None,
        branch2_fixed_k0=None,
        branch2_fixed_kinf=None,
        branch2_fixed_k_residues=(None,),
        branch2_fixed_f_poles_hz=(None,),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVariableMapperDimension:
    def test_dimension_1cell_movable_each_branch(self) -> None:
        """1 cell each branch, movable pole: 2*(logkm + fp) = 4 variables."""
        m = _simple_mapper_1cell()
        # b1: logkm_0, fp_0;  b2: logkm_0, fp_0
        assert m.dimension == 4

    def test_dimension_0cells_no_endpoints(self) -> None:
        """Zero cells, no endpoints → dimension 0."""
        m = build_variable_mapper(
            branch1_n_cells=0, branch1_has_c0=False, branch1_has_linf=False,
            branch1_pole_regions=(), branch1_k_box_bounds=(),
            branch1_k0_bounds=None, branch1_kinf_bounds=None,
            branch1_fixed_k0=None, branch1_fixed_kinf=None,
            branch1_fixed_k_residues=(), branch1_fixed_f_poles_hz=(),
            branch2_n_cells=0, branch2_has_c0=False, branch2_has_linf=False,
            branch2_pole_regions=(), branch2_k_box_bounds=(),
            branch2_k0_bounds=None, branch2_kinf_bounds=None,
            branch2_fixed_k0=None, branch2_fixed_kinf=None,
            branch2_fixed_k_residues=(), branch2_fixed_f_poles_hz=(),
        )
        assert m.dimension == 0

    def test_dimension_with_c0_and_linf(self) -> None:
        """Endpoints add 2 extra variables."""
        m = build_variable_mapper(
            branch1_n_cells=1, branch1_has_c0=True, branch1_has_linf=True,
            branch1_pole_regions=((1e6, 10e6),), branch1_k_box_bounds=((1.0, 1e4),),
            branch1_k0_bounds=(1.0, 1e4), branch1_kinf_bounds=(1e-9, 1e-3),
            branch1_fixed_k0=None, branch1_fixed_kinf=None,
            branch1_fixed_k_residues=(None,), branch1_fixed_f_poles_hz=(None,),
            branch2_n_cells=0, branch2_has_c0=False, branch2_has_linf=False,
            branch2_pole_regions=(), branch2_k_box_bounds=(),
            branch2_k0_bounds=None, branch2_kinf_bounds=None,
            branch2_fixed_k0=None, branch2_fixed_kinf=None,
            branch2_fixed_k_residues=(), branch2_fixed_f_poles_hz=(),
        )
        # b1: logk0 + logkinf + logkm_0 + fp_0 = 4; b2: 0
        assert m.dimension == 4

    def test_fixed_pole_not_in_dimension(self) -> None:
        """A FIXED pole (f_lo == f_hi) adds 0 variables for fp."""
        m = build_variable_mapper(
            branch1_n_cells=1, branch1_has_c0=False, branch1_has_linf=False,
            branch1_pole_regions=((5e6, 5e6),),  # FIXED: point interval
            branch1_k_box_bounds=((1.0, 1e4),),
            branch1_k0_bounds=None, branch1_kinf_bounds=None,
            branch1_fixed_k0=None, branch1_fixed_kinf=None,
            branch1_fixed_k_residues=(None,),
            branch1_fixed_f_poles_hz=(5e6,),    # FIXED pole value
            branch2_n_cells=0, branch2_has_c0=False, branch2_has_linf=False,
            branch2_pole_regions=(), branch2_k_box_bounds=(),
            branch2_k0_bounds=None, branch2_kinf_bounds=None,
            branch2_fixed_k0=None, branch2_fixed_kinf=None,
            branch2_fixed_k_residues=(), branch2_fixed_f_poles_hz=(),
        )
        # b1: logkm_0 only (fp is fixed so fp variable excluded)
        assert m.dimension == 1


class TestPackUnpackRoundtrip:
    def test_pack_unpack_roundtrip(self) -> None:
        """pack then unpack should recover the original physical values."""
        m = _simple_mapper_1cell()

        k_m_b1 = 150.0
        f_p_b1 = 4e6
        k_m_b2 = 800.0
        f_p_b2 = 20e6

        x = m.pack(
            k0_b1=None, k_inf_b1=None,
            k_residues_b1=(k_m_b1,), f_poles_b1=(f_p_b1,),
            k0_b2=None, k_inf_b2=None,
            k_residues_b2=(k_m_b2,), f_poles_b2=(f_p_b2,),
        )
        assert x.shape == (4,)
        assert np.all((x >= 0.0) & (x <= 1.0))

        b1, b2 = m.unpack(x)
        assert abs(b1.k_residues[0] - k_m_b1) / k_m_b1 < 1e-10
        assert abs(b1.f_poles_hz[0] - f_p_b1) / f_p_b1 < 1e-10
        assert abs(b2.k_residues[0] - k_m_b2) / k_m_b2 < 1e-10
        assert abs(b2.f_poles_hz[0] - f_p_b2) / f_p_b2 < 1e-10

    def test_l_c_derivation(self) -> None:
        """L_m and C_m should be correctly derived from k_m and f_p."""
        m = _simple_mapper_1cell()
        k_m = 200.0
        f_p = 3e6

        x = m.pack(
            k0_b1=None, k_inf_b1=None,
            k_residues_b1=(k_m,), f_poles_b1=(f_p,),
            k0_b2=None, k_inf_b2=None,
            k_residues_b2=(50.0,), f_poles_b2=(10e6,),
        )
        b1, _ = m.unpack(x)
        q_m = (_TWO_PI * f_p) ** 2
        expected_l = k_m / q_m
        expected_c = 1.0 / k_m
        assert abs(b1.l_values_h[0] - expected_l) / expected_l < 1e-9
        assert abs(b1.c_values_f[0] - expected_c) / expected_c < 1e-9

    def test_x_at_box_boundary(self) -> None:
        """x=0 and x=1 should return k at min and max of box."""
        m = _simple_mapper_1cell()

        # x=0 → min end
        x0 = np.zeros(m.dimension)
        b1_lo, _ = m.unpack(x0)
        assert b1_lo.k_residues[0] == pytest.approx(1.0, rel=1e-9)

        # x=1 → max end
        x1 = np.ones(m.dimension)
        b1_hi, _ = m.unpack(x1)
        assert b1_hi.k_residues[0] == pytest.approx(1e4, rel=1e-9)

    def test_zero_dim_pack_returns_empty(self) -> None:
        """Zero-dim mapper returns empty array."""
        m = build_variable_mapper(
            branch1_n_cells=0, branch1_has_c0=False, branch1_has_linf=False,
            branch1_pole_regions=(), branch1_k_box_bounds=(),
            branch1_k0_bounds=None, branch1_kinf_bounds=None,
            branch1_fixed_k0=None, branch1_fixed_kinf=None,
            branch1_fixed_k_residues=(), branch1_fixed_f_poles_hz=(),
            branch2_n_cells=0, branch2_has_c0=False, branch2_has_linf=False,
            branch2_pole_regions=(), branch2_k_box_bounds=(),
            branch2_k0_bounds=None, branch2_kinf_bounds=None,
            branch2_fixed_k0=None, branch2_fixed_kinf=None,
            branch2_fixed_k_residues=(), branch2_fixed_f_poles_hz=(),
        )
        x = m.pack(k0_b1=None, k_inf_b1=None, k_residues_b1=(),
                   f_poles_b1=(), k0_b2=None, k_inf_b2=None,
                   k_residues_b2=(), f_poles_b2=())
        assert x.shape == (0,)

    def test_pack_clips_to_unit_box(self) -> None:
        """Out-of-range physical values are clipped to [0,1]."""
        m = _simple_mapper_1cell()
        # k_m below box min → should clip to 0
        x = m.pack(
            k0_b1=None, k_inf_b1=None,
            k_residues_b1=(0.0001,), f_poles_b1=(4e6,),
            k0_b2=None, k_inf_b2=None,
            k_residues_b2=(1e6,), f_poles_b2=(20e6,),
        )
        assert np.all(x >= 0.0) and np.all(x <= 1.0)
