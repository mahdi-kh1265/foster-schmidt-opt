import math
import numpy as np
import pytest
from foster_eom.optimize.variable_map import VariableDescriptor
from foster_eom.sensitivities.foster_mapping import (
    dfp_dx, dC0_dx, dLinf_dx, dCm_dxkm, dLm_dxkm, dLm_dxfp
)

def test_dfp_dx():
    desc = VariableDescriptor("b1_fp_0", branch=1, var_type="fp", cell_index=0, f_lo_hz=1e6, f_hi_hz=5e6)
    assert dfp_dx(desc) == pytest.approx(4e6)

def test_dC0_dx_fd():
    k_min = 1e9
    k_max = 1e11
    log_range = math.log(k_max / k_min)
    desc = VariableDescriptor("b1_logk0", branch=1, var_type="logk0", log_k_box_min=math.log(k_min), log_k_box_range=log_range)
    
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
    desc = VariableDescriptor("b1_logkinf", branch=1, var_type="logkinf", log_k_box_min=math.log(k_min), log_k_box_range=log_range)
    
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
    desc_k = VariableDescriptor("b1_logkm_0", branch=1, var_type="logkm", log_k_box_min=math.log(k_min), log_k_box_range=log_range)
    desc_f = VariableDescriptor("b1_fp_0", branch=1, var_type="fp", cell_index=0, f_lo_hz=1e6, f_hi_hz=10e6)
    
    x_k = 0.5
    x_f = 0.3
    
    km = math.exp(desc_k.log_k_box_min + x_k * log_range)
    fp = desc_f.f_lo_hz + x_f * (desc_f.f_hi_hz - desc_f.f_lo_hz)
    
    Cm = 1.0 / km
    Lm = km / (2 * math.pi * fp)**2
    
    exact_dCm_dxkm = dCm_dxkm(Cm, desc_k)
    exact_dLm_dxkm = dLm_dxkm(Lm, desc_k)
    exact_dLm_dxfp = dLm_dxfp(Lm, fp, desc_f)
    
    h = 1e-5
    
    # FD for km
    km_plus = math.exp(desc_k.log_k_box_min + (x_k + h) * log_range)
    km_minus = math.exp(desc_k.log_k_box_min + (x_k - h) * log_range)
    fd_dCm_dxkm = (1.0 / km_plus - 1.0 / km_minus) / (2 * h)
    fd_dLm_dxkm = (km_plus / (2 * math.pi * fp)**2 - km_minus / (2 * math.pi * fp)**2) / (2 * h)
    
    assert exact_dCm_dxkm == pytest.approx(fd_dCm_dxkm, rel=1e-6)
    assert exact_dLm_dxkm == pytest.approx(fd_dLm_dxkm, rel=1e-6)
    
    # FD for fp
    fp_plus = desc_f.f_lo_hz + (x_f + h) * (desc_f.f_hi_hz - desc_f.f_lo_hz)
    fp_minus = desc_f.f_lo_hz + (x_f - h) * (desc_f.f_hi_hz - desc_f.f_lo_hz)
    fd_dLm_dxfp = (km / (2 * math.pi * fp_plus)**2 - km / (2 * math.pi * fp_minus)**2) / (2 * h)
    
    assert exact_dLm_dxfp == pytest.approx(fd_dLm_dxfp, rel=1e-5)
