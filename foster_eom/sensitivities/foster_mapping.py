"""Analytical derivatives for the Foster coordinate mapping."""

from foster_eom.optimize.variable_map import VariableDescriptor


def dfp_dx(desc: VariableDescriptor) -> float:
    """Derivative of pole frequency f_p w.r.t its normalized coordinate x."""
    assert desc.f_lo_hz is not None and desc.f_hi_hz is not None
    span = desc.f_hi_hz - desc.f_lo_hz
    return span if span > 0.0 else 0.0


def dC0_dx(c0_f: float, desc: VariableDescriptor) -> float:
    """Derivative of endpoint capacitor C_0 w.r.t its normalized log-k coordinate x."""
    assert desc.log_k_box_range is not None
    return -c0_f * desc.log_k_box_range


def dLinf_dx(l_inf_h: float, desc: VariableDescriptor) -> float:
    """Derivative of endpoint inductor L_inf w.r.t its normalized log-k coordinate x."""
    assert desc.log_k_box_range is not None
    return l_inf_h * desc.log_k_box_range


def dCm_dxkm(cm_f: float, desc: VariableDescriptor) -> float:
    """Derivative of cell capacitor C_m w.r.t its normalized log-k coordinate x."""
    assert desc.log_k_box_range is not None
    return -cm_f * desc.log_k_box_range


def dLm_dxkm(lm_h: float, desc: VariableDescriptor) -> float:
    """Derivative of cell inductor L_m w.r.t its normalized log-k coordinate x."""
    assert desc.log_k_box_range is not None
    return lm_h * desc.log_k_box_range


def dLm_dxfp(lm_h: float, fp_hz: float, desc_fp: VariableDescriptor) -> float:
    """Derivative of cell inductor L_m w.r.t its normalized pole coordinate x."""
    assert desc_fp.f_lo_hz is not None and desc_fp.f_hi_hz is not None
    span = desc_fp.f_hi_hz - desc_fp.f_lo_hz
    if span <= 0.0 or fp_hz <= 0.0:
        return 0.0
    return -2.0 * lm_h * span / fp_hz
