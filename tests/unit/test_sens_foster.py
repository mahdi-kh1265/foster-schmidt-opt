import math

import numpy as np
import pytest

from foster_eom.optimize.variable_map import VariableDescriptor
from foster_eom.sensitivities.foster_mapping import (
    dC0_dx,
    dCm_dxkm,
    dfp_dx,
    dLinf_dx,
    dLm_dxfp,
    dLm_dxkm,
)


def test_dfp_dx():
    desc = VariableDescriptor(
        "b1_fp_0", branch=1, var_type="fp", cell_index=0, f_lo_hz=1e6, f_hi_hz=5e6
    )
    assert dfp_dx(desc) == pytest.approx(4e6)


def test_dC0_dx_fd():
    k_min = 1e9
    k_max = 1e11
    log_range = math.log(k_max / k_min)
    desc = VariableDescriptor(
        "b1_logk0",
        branch=1,
        var_type="logk0",
        log_k_box_min=math.log(k_min),
        log_k_box_range=log_range,
    )

    x = 0.5
    k0 = math.exp(desc.log_k_box_min + x * log_range)
    C0 = 1.0 / k0
    exact = dC0_dx(C0, desc)

    h = 1e-5
    k0_plus = math.exp(desc.log_k_box_min + (x + h) * log_range)
    k0_minus = math.exp(desc.log_k_box_min + (x - h) * log_range)
    fd = (1.0 / k0_plus - 1.0 / k0_minus) / (2 * h)

    assert exact == pytest.approx(fd, rel=1e-6)


def test_dLinf_dx_fd():
    k_min = 1e-9
    k_max = 1e-7
    log_range = math.log(k_max / k_min)
    desc = VariableDescriptor(
        "b1_logkinf",
        branch=1,
        var_type="logkinf",
        log_k_box_min=math.log(k_min),
        log_k_box_range=log_range,
    )

    x = 0.5
    Linf = math.exp(desc.log_k_box_min + x * log_range)
    exact = dLinf_dx(Linf, desc)

    h = 1e-5
    L_plus = math.exp(desc.log_k_box_min + (x + h) * log_range)
    L_minus = math.exp(desc.log_k_box_min + (x - h) * log_range)
    fd = (L_plus - L_minus) / (2 * h)

    assert exact == pytest.approx(fd, rel=1e-6)


def test_cell_components_fd():
    k_min = 1e9
    k_max = 1e12
    log_range = math.log(k_max / k_min)
    desc_k = VariableDescriptor(
        "b1_logkm_0",
        branch=1,
        var_type="logkm",
        log_k_box_min=math.log(k_min),
        log_k_box_range=log_range,
    )
    desc_f = VariableDescriptor(
        "b1_fp_0", branch=1, var_type="fp", cell_index=0, f_lo_hz=1e6, f_hi_hz=10e6
    )

    x_k = 0.5
    x_f = 0.3

    km = math.exp(desc_k.log_k_box_min + x_k * log_range)
    fp = desc_f.f_lo_hz + x_f * (desc_f.f_hi_hz - desc_f.f_lo_hz)

    Cm = 1.0 / km
    Lm = km / (2 * math.pi * fp) ** 2

    exact_dCm_dxkm = dCm_dxkm(Cm, desc_k)
    exact_dLm_dxkm = dLm_dxkm(Lm, desc_k)
    exact_dLm_dxfp = dLm_dxfp(Lm, fp, desc_f)

    h = 1e-5

    # FD for km
    km_plus = math.exp(desc_k.log_k_box_min + (x_k + h) * log_range)
    km_minus = math.exp(desc_k.log_k_box_min + (x_k - h) * log_range)
    fd_dCm_dxkm = (1.0 / km_plus - 1.0 / km_minus) / (2 * h)
    fd_dLm_dxkm = (km_plus / (2 * math.pi * fp) ** 2 - km_minus / (2 * math.pi * fp) ** 2) / (2 * h)

    assert exact_dCm_dxkm == pytest.approx(fd_dCm_dxkm, rel=1e-6)
    assert exact_dLm_dxkm == pytest.approx(fd_dLm_dxkm, rel=1e-6)

    # FD for fp
    fp_plus = desc_f.f_lo_hz + (x_f + h) * (desc_f.f_hi_hz - desc_f.f_lo_hz)
    fp_minus = desc_f.f_lo_hz + (x_f - h) * (desc_f.f_hi_hz - desc_f.f_lo_hz)
    fd_dLm_dxfp = (km / (2 * math.pi * fp_plus) ** 2 - km / (2 * math.pi * fp_minus) ** 2) / (2 * h)

    assert exact_dLm_dxfp == pytest.approx(fd_dLm_dxfp, rel=1e-5)


def test_end_to_end_decision_variable_mapper_fd():
    from foster_eom.optimize.variable_map import build_variable_mapper

    mapper = build_variable_mapper(
        branch1_n_cells=1,
        branch1_has_c0=True,
        branch1_has_linf=True,
        branch1_pole_regions=((1e6, 10e6),),
        branch1_k_box_bounds=((1e9, 1e12),),
        branch1_k0_bounds=(1e9, 1e12),
        branch1_kinf_bounds=(1e9, 1e12),
        branch1_fixed_k0=None,
        branch1_fixed_kinf=None,
        branch1_fixed_k_residues=(None,),
        branch1_fixed_f_poles_hz=(None,),
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

    # x array layout: logk0, logkinf, logkm_0, fp_0
    x_val = np.array([0.5, 0.5, 0.5, 0.3])
    phys_b1, _ = mapper.unpack(x_val)

    # The cells are at index 0. C0 is added after cells, so c_values_f[1]. Linf is l_values_h[1].
    C0_val = phys_b1.c_values_f[1]
    Linf_val = phys_b1.l_values_h[1]
    Cm_val = phys_b1.c_values_f[0]
    Lm_val = phys_b1.l_values_h[0]
    fp_val = phys_b1.f_poles_hz[0]

    exact_dC0_dx = dC0_dx(C0_val, mapper.descriptors[0])
    exact_dLinf_dx = dLinf_dx(Linf_val, mapper.descriptors[1])
    exact_dCm_dxkm = dCm_dxkm(Cm_val, mapper.descriptors[2])
    exact_dLm_dxkm = dLm_dxkm(Lm_val, mapper.descriptors[2])
    exact_dLm_dxfp = dLm_dxfp(Lm_val, fp_val, mapper.descriptors[3])

    h = 1e-5

    # dC0/dx
    x_plus = x_val.copy()
    x_plus[0] += h
    x_minus = x_val.copy()
    x_minus[0] -= h
    b1_plus, _ = mapper.unpack(x_plus)
    b1_minus, _ = mapper.unpack(x_minus)
    fd_dC0_dx = (b1_plus.c_values_f[1] - b1_minus.c_values_f[1]) / (2 * h)
    assert exact_dC0_dx == pytest.approx(fd_dC0_dx, rel=1e-6)

    # dLinf/dx
    x_plus = x_val.copy()
    x_plus[1] += h
    x_minus = x_val.copy()
    x_minus[1] -= h
    b1_plus, _ = mapper.unpack(x_plus)
    b1_minus, _ = mapper.unpack(x_minus)
    fd_dLinf_dx = (b1_plus.l_values_h[1] - b1_minus.l_values_h[1]) / (2 * h)
    assert exact_dLinf_dx == pytest.approx(fd_dLinf_dx, rel=1e-6)

    # dCm/dxkm and dLm/dxkm
    x_plus = x_val.copy()
    x_plus[2] += h
    x_minus = x_val.copy()
    x_minus[2] -= h
    b1_plus, _ = mapper.unpack(x_plus)
    b1_minus, _ = mapper.unpack(x_minus)
    fd_dCm_dxkm = (b1_plus.c_values_f[0] - b1_minus.c_values_f[0]) / (2 * h)
    fd_dLm_dxkm = (b1_plus.l_values_h[0] - b1_minus.l_values_h[0]) / (2 * h)
    assert exact_dCm_dxkm == pytest.approx(fd_dCm_dxkm, rel=1e-6)
    assert exact_dLm_dxkm == pytest.approx(fd_dLm_dxkm, rel=1e-6)

    # dLm/dxfp
    x_plus = x_val.copy()
    x_plus[3] += h
    x_minus = x_val.copy()
    x_minus[3] -= h
    b1_plus, _ = mapper.unpack(x_plus)
    b1_minus, _ = mapper.unpack(x_minus)
    fd_dLm_dxfp = (b1_plus.l_values_h[0] - b1_minus.l_values_h[0]) / (2 * h)
    assert exact_dLm_dxfp == pytest.approx(fd_dLm_dxfp, rel=1e-6)
