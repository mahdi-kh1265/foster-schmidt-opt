import numpy as np

from foster_eom.circuit.measurements import CircuitSolution
from foster_eom.optimize.constraints import ConstraintDescriptor, ConstraintLayout
from foster_eom.optimize.variable_map import DecisionVariableMapper
from foster_eom.sensitivities.foster_mapping import (
    dCm_dxkm,
    dfp_dx,
    dLm_dxfp,
    dLm_dxkm,
)
from foster_eom.sensitivities.observables import ObservableDerivatives


def _get_coordinate_gradients(
    desc_type: str,
    branch: int,
    m: int,
    mapper: DecisionVariableMapper,
    x: np.ndarray,
) -> np.ndarray:
    """Compute gradient of coordinate-only constraint wrt normalized vector x."""
    grad = np.zeros(mapper.dimension, dtype=np.float64)
    b1, b2 = mapper.unpack(x)
    phys = b1 if branch == 1 else b2

    idx_km = -1
    idx_fp = -1
    idx_fp_next = -1

    for i, d in enumerate(mapper.descriptors):
        if d.branch == branch:
            if d.var_type == "logkm" and d.cell_index == m:
                idx_km = i
            elif d.var_type == "fp":
                if d.cell_index == m:
                    idx_fp = i
                elif d.cell_index == m + 1:
                    idx_fp_next = i

    if desc_type in ("comp_L_hi", "comp_L_lo"):
        sign = -1.0 if desc_type == "comp_L_hi" else 1.0
        if m < len(phys.l_values_h):
            L_m = phys.l_values_h[m]
            if idx_km != -1:
                grad[idx_km] = sign * dLm_dxkm(L_m, mapper.descriptors[idx_km])
            if idx_fp != -1:
                fp = phys.f_poles_hz[m]
                grad[idx_fp] = sign * dLm_dxfp(L_m, fp, mapper.descriptors[idx_fp])

    elif desc_type in ("comp_C_hi", "comp_C_lo"):
        sign = -1.0 if desc_type == "comp_C_hi" else 1.0
        if m < len(phys.c_values_f):
            C_m = phys.c_values_f[m]
            if idx_km != -1:
                grad[idx_km] = sign * dCm_dxkm(C_m, mapper.descriptors[idx_km])

    elif desc_type == "pole_sep":
        if idx_fp_next != -1:
            grad[idx_fp_next] = dfp_dx(mapper.descriptors[idx_fp_next])
        if idx_fp != -1:
            grad[idx_fp] = -dfp_dx(mapper.descriptors[idx_fp])

    return grad


def compute_constraint_jacobian_row(
    desc: ConstraintDescriptor,
    sol: CircuitSolution | None,
    obs: ObservableDerivatives | None,
    mapper: DecisionVariableMapper,
    x: np.ndarray,
    l_max: float,
    c_max: float,
    sep_b1: float,
    sep_b2: float,
) -> np.ndarray:
    """Compute the parameter gradient of a single constraint descriptor."""
    grad = np.zeros(mapper.dimension, dtype=np.float64)
    ct = desc.constraint_type
    scale = desc.normalization_scale

    if ct == "gamma":
        if sol is not None and obs is not None and sol.gamma is not None:
            gamma = sol.gamma
            abs_g = abs(gamma)
            if abs_g > 0:
                dabs = np.real(np.conj(gamma) * obs.gamma) / abs_g
                grad = -dabs / scale
    elif ct == "r_max":
        if sol is not None and obs is not None and sol.z_in is not None:
            grad = -np.real(obs.z_in) / scale
    elif ct == "r_min":
        if sol is not None and obs is not None and sol.z_in is not None:
            grad = np.real(obs.z_in) / scale
    elif ct == "x_bound":
        if sol is not None and obs is not None and sol.z_in is not None:
            x_in = sol.z_in.imag
            dx_in = np.imag(obs.z_in)
            sign_x = 1.0 if x_in >= 0 else -1.0
            dabs = sign_x * dx_in
            grad = -dabs / scale
    elif ct == "i_source":
        if sol is not None and obs is not None and sol.i_source_droop is not None:
            i_s = sol.i_source_droop
            abs_i = abs(i_s)
            if abs_i > 0:
                dabs = np.real(np.conj(i_s) * obs.i_port) / abs_i
                grad = -dabs / scale
    elif ct in ("comp_L_hi", "comp_L_lo", "comp_C_hi", "comp_C_lo", "pole_sep"):
        b = desc.branch
        m = desc.cell_index
        if b is not None and m is not None:
            raw_grad = _get_coordinate_gradients(ct, b, m, mapper, x)
            if ct in ("comp_L_hi", "comp_L_lo"):
                grad = raw_grad / max(l_max, 1e-20)
            elif ct in ("comp_C_hi", "comp_C_lo"):
                grad = raw_grad / max(c_max, 1e-20)
            elif ct == "pole_sep":
                sep = sep_b1 if b == 1 else sep_b2
                grad = raw_grad / max(sep, 1.0)

    return grad


def compute_layout_jacobian(
    layout: ConstraintLayout,
    target_solutions: dict[int, CircuitSolution],
    target_observables: dict[int, ObservableDerivatives],
    off_target_gradients: dict[int, np.ndarray],
    mapper: DecisionVariableMapper,
    x: np.ndarray,
    l_max: float,
    c_max: float,
    sep_b1: float,
    sep_b2: float,
) -> np.ndarray:
    """Compute the full ConstraintLayout Jacobian matrix.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_constraints, n_params).
    """
    n_c = layout.n
    J = np.zeros((n_c, mapper.dimension), dtype=np.float64)

    for i, desc in enumerate(layout.descriptors):
        ct = desc.constraint_type
        if ct in ("offtarget", "v_min", "v_max"):
            if desc.freq_index is not None and desc.freq_index in off_target_gradients:
                dabs = off_target_gradients[desc.freq_index]
                if ct in ("offtarget", "v_max"):
                    J[i, :] = -dabs / desc.normalization_scale
                elif ct == "v_min":
                    J[i, :] = dabs / desc.normalization_scale
            continue

        sol = None
        obs = None
        if desc.freq_index is not None and desc.freq_index in target_solutions:
            sol = target_solutions[desc.freq_index]
            obs = target_observables.get(desc.freq_index)

        J[i, :] = compute_constraint_jacobian_row(
            desc, sol, obs, mapper, x, l_max, c_max, sep_b1, sep_b2
        )

    return J
