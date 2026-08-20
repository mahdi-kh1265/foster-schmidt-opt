"""Validates coordinate-only constraint derivatives against central FD
of production ConstraintLayout.evaluate().

Coordinate-only derivative evaluation invokes zero MNA solves.
"""

import numpy as np

from foster_eom.domain.constraints import (
    ConstraintSeverity,
    FrequencyScope,
    MatchConstraints,
    StressConstraints,
)
from foster_eom.optimize.constraints import (
    ConstraintDescriptor,
    compile_constraint_layout,
)
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.sensitivities.constraints import (
    compute_constraint_jacobian_row,
)


def _build_test_mapper():
    """Build a 2-cell branch-1 mapper with movable poles and no endpoints."""
    return build_variable_mapper(
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
        branch2_fixed_f_poles_hz=(),
    )


def _eval_coord_constraints(mapper, x, l_max, c_max, sep_b1, sep_b2):
    """Evaluate production ConstraintLayout for coordinate-only constraints.

    Returns the constraint margin vector from production evaluate().
    """
    b1, b2 = mapper.unpack(x)

    # Build a layout with component-bound + pole-sep constraints
    match_c = MatchConstraints(gamma_max=0.5, resistance_max_ohm=50.0)
    stress_c = StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=2.0)

    layout = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=(),
        evaluation_frequencies_hz=(),
        target_indices=(),
        off_target_indices=(),
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=2,
        n_cells_b2=0,
        z_ref_ohm=50.0,
    )

    g = layout.evaluate(
        solutions=(),
        target_indices=(),
        off_target_indices=(),
        branch1_pole_regions=((1e6, 5e6), (6e6, 10e6)),
        branch2_pole_regions=(),
        branch1_k_residues=b1.k_residues,
        branch2_k_residues=b2.k_residues,
        branch1_f_poles=b1.f_poles_hz,
        branch2_f_poles=b2.f_poles_hz,
        branch1_l_vals=b1.l_values_h,
        branch2_l_vals=b2.l_values_h,
        branch1_c_vals=b1.c_values_f,
        branch2_c_vals=b2.c_values_f,
        component_limits_l_min=1e-9,
        component_limits_l_max=l_max,
        component_limits_c_min=1e-12,
        component_limits_c_max=c_max,
        pole_sep_min_b1=sep_b1,
        pole_sep_min_b2=sep_b2,
        z_ref_ohm=50.0,
        gamma_max=0.5,
        r_min_ohm=35.0,
        r_max_ohm=70.0,
        x_max_ohm=20.0,
        source_current_max_a=1.0,
        off_target_eom_peak_rms_v=2.0,
    )
    return layout, g


def test_coordinate_constraint_derivatives():
    """Validates coordinate-only constraint derivatives against central FD
    of production ConstraintLayout.evaluate().

    Covers: L bounds, C bounds, pole separation.
    Verifies zero MNA solves (pure coordinate derivatives).
    """
    mapper = _build_test_mapper()
    x_val = np.array([0.5, 0.4, 0.6, 0.7])  # logkm_0, fp_0, logkm_1, fp_1

    l_max = 1e-3
    c_max = 1e-6
    sep_b1 = 1e6
    sep_b2 = 1e3

    # Get production constraint evaluation at x_val
    layout, _g_nom = _eval_coord_constraints(mapper, x_val, l_max, c_max, sep_b1, sep_b2)

    # Compute analytical Jacobian
    J_ana = np.zeros((layout.n, mapper.dimension), dtype=np.float64)
    for i, desc in enumerate(layout.descriptors):
        J_ana[i, :] = compute_constraint_jacobian_row(
            desc, None, None, mapper, x_val, l_max, c_max, sep_b1, sep_b2
        )

    # Central FD of production evaluate
    h = 1e-7
    J_fd = np.zeros((layout.n, mapper.dimension), dtype=np.float64)
    for k in range(mapper.dimension):
        x_plus = x_val.copy()
        x_minus = x_val.copy()
        x_plus[k] += h
        x_minus[k] -= h
        _, g_plus = _eval_coord_constraints(mapper, x_plus, l_max, c_max, sep_b1, sep_b2)
        _, g_minus = _eval_coord_constraints(mapper, x_minus, l_max, c_max, sep_b1, sep_b2)
        J_fd[:, k] = (g_plus - g_minus) / (2 * h)

    # Validate each row
    for i, desc in enumerate(layout.descriptors):
        ct = desc.constraint_type
        for k in range(mapper.dimension):
            if abs(J_fd[i, k]) > 1e-15 or abs(J_ana[i, k]) > 1e-15:
                np.testing.assert_allclose(
                    J_ana[i, k],
                    J_fd[i, k],
                    rtol=1e-5,
                    atol=1e-12,
                    err_msg=f"Row {i} ({ct}, branch={desc.branch}, cell={desc.cell_index}), param {k}",
                )


def test_coordinate_constraint_zero_mna_solves():
    """Verify that coordinate-only constraint derivatives invoke zero MNA solves."""
    mapper = _build_test_mapper()
    x_val = np.array([0.5, 0.4, 0.6, 0.7])

    desc = ConstraintDescriptor(
        "L_hi",
        "comp_L_hi",
        FrequencyScope.ALL_TARGETS,
        ConstraintSeverity.HARD,
        branch=1,
        cell_index=0,
    )
    # compute_constraint_jacobian_row with sol=None, obs=None should work
    # (no MNA data needed for coordinate constraints)
    grad = compute_constraint_jacobian_row(
        desc,
        None,
        None,
        mapper,
        x_val,
        l_max=1e-3,
        c_max=1e-6,
        sep_b1=1e6,
        sep_b2=1e3,
    )
    assert grad.shape == (mapper.dimension,)
    # Should have nonzero entries for logkm_0 and fp_0 only
    assert grad[0] != 0.0  # logkm_0 affects L_0
    assert grad[1] != 0.0  # fp_0 affects L_0
