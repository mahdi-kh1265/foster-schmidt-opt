import math
import numpy as np
import pytest

from foster_eom.optimize.constraints import compile_constraint_layout, ConstraintDescriptor
from foster_eom.domain.constraints import MatchConstraints, StressConstraints, ConstraintSeverity, FrequencyScope
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.sensitivities.constraints import compute_constraint_jacobian_row, compute_layout_jacobian

def test_coordinate_constraint_derivatives():
    """Validates the coordinate-only constraint derivatives against FD."""
    mapper = build_variable_mapper(
        branch1_n_cells=2,
        branch1_has_c0=False,
        branch1_has_linf=False,
        branch1_pole_regions=((1e6, 5e6), (6e6, 10e6)),
        branch1_k_box_bounds=((1e9, 1e12), (1e9, 1e12)),
        branch1_k0_bounds=None,
        branch1_kinf_bounds=None,
        branch1_fixed_k0=None,
        branch1_fixed_kinf=None,
        branch1_fixed_k_residues=(None, None),
        branch1_fixed_f_poles_hz=(None, None),
        branch2_n_cells=0,
        branch2_has_c0=False,
        branch2_has_linf=False,
        branch2_pole_regions=(),
        branch2_k_box_bounds=(),
        branch2_k0_bounds=None,
        branch2_kinf_bounds=None,
        branch2_fixed_k0=None,
        branch2_fixed_kinf=None,
        branch2_fixed_k_residues=(),
        branch2_fixed_f_poles_hz=()
    )
    
    x_val = np.array([0.5, 0.4, 0.6, 0.7])  # logkm_0, fp_0, logkm_1, fp_1
    
    # comp_L_hi: L_0 < 1e-6
    desc = ConstraintDescriptor("L_hi", "comp_L_hi", FrequencyScope.ALL_TARGETS, ConstraintSeverity.HARD, branch=1, cell_index=0, normalization_scale=2.0)
    
    grad = compute_constraint_jacobian_row(desc, None, None, mapper, x_val)
    
    h = 1e-6
    x_plus = x_val.copy()
    x_plus[0] += h
    b1_plus, _ = mapper.unpack(x_plus)
    g_plus = (1e-6 - b1_plus.l_values_h[0]) / 2.0
    
    x_minus = x_val.copy()
    x_minus[0] -= h
    b1_minus, _ = mapper.unpack(x_minus)
    g_minus = (1e-6 - b1_minus.l_values_h[0]) / 2.0
    
    fd_0 = (g_plus - g_minus) / (2 * h)
    assert grad[0] == pytest.approx(fd_0, rel=1e-5)
    
    # pole_sep: fp_1 - fp_0 > sep
    desc_sep = ConstraintDescriptor("sep", "pole_sep", FrequencyScope.ALL_TARGETS, ConstraintSeverity.HARD, branch=1, cell_index=0, normalization_scale=1.0)
    
    grad_sep = compute_constraint_jacobian_row(desc_sep, None, None, mapper, x_val)
    
    # FD for fp_0
    x_plus[1] += h
    x_plus[0] = x_val[0]
    b1_plus, _ = mapper.unpack(x_plus)
    g_plus = (b1_plus.f_poles_hz[1] - b1_plus.f_poles_hz[0] - 1e6) / 1.0
    
    x_minus[1] -= h
    x_minus[0] = x_val[0]
    b1_minus, _ = mapper.unpack(x_minus)
    g_minus = (b1_minus.f_poles_hz[1] - b1_minus.f_poles_hz[0] - 1e6) / 1.0
    
    fd_1 = (g_plus - g_minus) / (2 * h)
    assert grad_sep[1] == pytest.approx(fd_1, rel=1e-5)
