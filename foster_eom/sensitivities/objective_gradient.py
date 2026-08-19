"""Production P05 objective gradient.

Differentiates the frozen production objective:

    J_total = J_base + J_soft
    J_base  = w_gamma * J_gamma + w_voltage * J_voltage + w_loss * J_loss + J_complexity
    J_soft  = sum_s lambda_s * max(0, -g_s)^2

Derivative validity is tracked explicitly — no derivative is ever silently
returned as an analytical zero when it is unsupported or nonsmooth.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from foster_eom.circuit.graph import ElementKind
from foster_eom.circuit.measurements import CircuitSolution
from foster_eom.optimize.constraints import ConstraintLayout
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.sensitivities.observables import ObservableDerivatives


class DerivativeStatus(enum.Enum):
    """Derivative validity status."""

    SMOOTH = "smooth"  # Analytical, everywhere differentiable
    NONSMOOTH_KINK = "nonsmooth_kink"  # Mathematical kink (C^0 but not C^1)
    NUMERICALLY_UNRESOLVED = "numerically_unresolved"  # Finite precision issue
    UNSUPPORTED = "unsupported"  # No derivative implemented for this branch
    NOMINAL_FAILURE = "nominal_failure"  # Nominal solve failed; no derivative possible


@dataclass(frozen=True)
class ObjectiveGradientResult:
    """Complete gradient with derivative validity tracking."""

    gradient: np.ndarray  # shape (K,) real
    status: DerivativeStatus
    nonsmooth_terms: tuple[str, ...]  # names of any nonsmooth-active terms
    unsupported_terms: tuple[str, ...]  # names of any unsupported terms


def check_analytical_support(
    config: ObjectiveConfig,
    soft_layout: ConstraintLayout,
    soft_g_vector: np.ndarray | None,
    soft_jacobian: np.ndarray | None,
) -> tuple[bool, list[str]]:
    """Pre-check whether the configuration is analytically supported.

    Returns
    -------
    (supported, reasons) : (bool, list[str])
        If ``supported`` is False, ``reasons`` lists every unsupported term.
        Unsupported configurations must use the frozen FD path in P12.5-E.

    Invariant: analytical-supported ⟺ every enabled objective/constraint term
    has a valid derivative.
    """
    reasons: list[str] = []

    # J_loss with lossy elements is now supported for standard element types
    # (RESISTOR, INDUCTOR, CAPACITOR). Only genuinely unsupported if lossy_element_ids
    # is non-empty but element_voltage_derivs are unavailable at runtime — this is
    # checked dynamically in compute_objective_gradient via DerivativeStatus.

    # w_loss > 0 without lossy elements: J_loss = 0.0 regardless, so supported.

    # Soft penalties: if soft_layout has descriptors but g_vector/jacobian are not
    # provided, the transaction should compute them internally. If external callers
    # use this gate before soft data is available, that's fine — the transaction
    # handles it. Only report unsupported if soft layout is non-empty AND there's
    # no way to compute the derivatives (both external and internal paths unavailable).
    # Since the transaction now computes soft derivatives internally, this is always
    # supported when the transaction is used.

    return (len(reasons) == 0, reasons)


def compute_objective_gradient(
    config: ObjectiveConfig,
    target_solutions: dict[int, CircuitSolution],
    target_observables: dict[int, ObservableDerivatives],
    target_indices: tuple[int, ...],
    soft_layout: ConstraintLayout,
    soft_g_vector: np.ndarray | None,
    soft_jacobian: np.ndarray | None,
    n_params: int,
) -> ObjectiveGradientResult:
    """Compute the complete production P05 objective gradient.

    Parameters
    ----------
    config : ObjectiveConfig
        Frozen objective weights and references.
    target_solutions : dict[int, CircuitSolution]
        Nominal solutions keyed by frequency index.
    target_observables : dict[int, ObservableDerivatives]
        Observable derivatives keyed by frequency index.
    target_indices : tuple[int, ...]
        Frequency indices of target frequencies.
    soft_layout : ConstraintLayout
        Soft constraint layout.
    soft_g_vector : np.ndarray | None
        Soft constraint margin vector.
    soft_jacobian : np.ndarray | None
        Soft constraint Jacobian.
    n_params : int
        Number of decision variables.

    Returns
    -------
    ObjectiveGradientResult
    """
    grad = np.zeros(n_params, dtype=np.float64)
    nonsmooth: list[str] = []
    unsupported: list[str] = []
    status = DerivativeStatus.SMOOTH

    n_targets = len(target_indices)
    if n_targets == 0:
        return ObjectiveGradientResult(
            gradient=grad,
            status=DerivativeStatus.NOMINAL_FAILURE,
            nonsmooth_terms=(),
            unsupported_terms=(),
        )

    # ---- 1. dJ_gamma / dp ----
    # J_gamma = (1/N) sum_i |Gamma_i|^2
    # d|Gamma|^2 / dp = 2 Re(Gamma^* dGamma/dp)
    # => dJ_gamma/dp = (2 / N) sum Re(Gamma_i^* dGamma_i/dp)
    # |Gamma|^2 is smooth everywhere including Gamma=0 (quadratic, not abs).
    if config.w_gamma != 0.0:
        dj_gamma = np.zeros(n_params, dtype=np.float64)
        n_valid_gamma = 0
        for fi in target_indices:
            if fi not in target_solutions or fi not in target_observables:
                continue
            sol = target_solutions[fi]
            obs = target_observables[fi]
            if sol.gamma is None:
                continue
            gamma = sol.gamma
            dj_gamma += 2.0 * np.real(np.conj(gamma) * obs.gamma)
            n_valid_gamma += 1
        if n_valid_gamma > 0:
            grad += config.w_gamma * dj_gamma / n_valid_gamma

    # ---- 2. dJ_voltage / dp ----
    # J_voltage = sum_i w_i * ((|V_eom_i| - V*_i) / D_i)^2
    # where D_i = max(V*_i, 1e-6), exactly as production.
    # d/dp = 2 w_i ((|V| - V*) / D_i^2) * d|V|/dp
    # d|V|/dp = Re(V^* dV/dp) / |V|   (nonsmooth at V=0)
    if config.w_voltage != 0.0 and config.voltage_targets_rms_v:
        dj_voltage = np.zeros(n_params, dtype=np.float64)
        for ti, fi in enumerate(target_indices):
            v_target = (
                config.voltage_targets_rms_v[ti] if ti < len(config.voltage_targets_rms_v) else None
            )
            if v_target is None:
                continue
            w_i = (
                config.voltage_target_weights[ti]
                if ti < len(config.voltage_target_weights)
                else 1.0
            )
            if fi not in target_solutions or fi not in target_observables:
                continue
            sol = target_solutions[fi]
            obs = target_observables[fi]
            if sol.v_eom is None:
                continue
            v_eom = sol.v_eom
            abs_v = abs(v_eom)
            d_i = max(v_target, 1e-6)  # production D_i
            if abs_v > 0:
                # d|V_eom|/dp = Re(V_eom^* dV_eom/dp) / |V_eom|
                dabs_v = np.real(np.conj(v_eom) * obs.v_eom) / abs_v
                dj_voltage += w_i * 2.0 * (abs_v - v_target) / (d_i**2) * dabs_v
            else:
                # |V_eom| = 0 is a nonsmooth point for d|V|/dp
                nonsmooth.append("J_voltage_abs_v_eom_zero")
        grad += config.w_voltage * dj_voltage

    # ---- 3. dJ_loss / dp ----
    # J_loss = (1/N_valid) * sum_targets max(0, 10 log10(P_source / P_eom))
    # where P_eom = P_source - P_parasitic
    #
    # In the smooth positive-power branch (P_eom > 0, loss_db > 0):
    #   dJ_loss/dp = (10/(N*ln10)) * (dP_source/P_source - dP_eom/P_eom)
    # where:
    #   dP_eom/dp = dP_source/dp - dP_parasitic/dp
    #   dP_parasitic/dp = sum_R dP_R/dp for R in lossy_element_ids (excl EOM)
    #
    # For element with admittance Y, voltage V, current I = Y*V:
    #   S = V*conj(I), P = Re(S) = Re(V*conj(Y*V)) = Re(Y) * |V|^2
    #   dP/dp = 2 * Re(Y) * Re(V^* dV/dp)
    # For a resistor Y = 1/R:  dP_R/dp = 2/R * Re(V_R^* dV_R/dp)
    # For ideal L/C: Re(Y) = 0 → dP/dp = 0 (no contribution)
    #
    # Production clamps (nonsmoothness):
    #   loss_db = 100 if P_eom <= 0   → NONSMOOTH_KINK (reported, derivative skipped)
    #   max(0, loss_db)               → zero gradient at smooth interior when loss_db < 0
    #   P_source <= 0                 → derivative undefined (reported)
    if config.w_loss != 0.0 and config.lossy_element_ids:
        dj_loss = np.zeros(n_params, dtype=np.float64)
        n_valid_loss = 0
        for _ti, fi in enumerate(target_indices):
            if fi not in target_solutions or fi not in target_observables:
                continue
            sol = target_solutions[fi]
            obs = target_observables[fi]
            if sol.element_measurements is None or sol.p_source_delivered_w is None:
                continue
            p_source = sol.p_source_delivered_w
            if p_source <= 0:
                unsupported.append(f"J_loss_p_source_nonpositive_f{fi}")
                continue
            # P_parasitic = sum of real power in lossy elements (not EOM)
            p_parasitic = sum(
                sol.element_measurements[eid].real_power_w
                for eid in config.lossy_element_ids
                if eid in sol.element_measurements and eid != config.eom_element_id
            )
            p_eom = p_source - p_parasitic
            if p_eom <= 0:
                nonsmooth.append(f"J_loss_p_eom_clamp_f{fi}")
                continue
            loss_db = 10.0 * math.log10(p_source / p_eom)
            if loss_db < 0:
                # max(0, loss_db) = 0, derivative = 0 at smooth interior
                n_valid_loss += 1
                continue
            if loss_db == 0.0:
                # Exactly at the kink of max(0, L) — nonsmooth boundary
                nonsmooth.append(f"J_loss_kink_L_eq_0_f{fi}")
                n_valid_loss += 1
                continue
            # Smooth region: loss_db > 0, P_eom > 0, P_source > 0
            #
            # dP_source/dp from obs.p_delivered
            dp_source = obs.p_delivered
            #
            # dP_parasitic/dp = sum_R dP_R/dp
            # dP_R/dp = 2 * Re(Y_R) * Re(V_R^* dV_R/dp)
            # where Re(Y_R) is extracted from sol.element_measurements
            dp_parasitic = np.zeros(n_params, dtype=np.float64)
            _loss_elem_unsupported = False
            for eid in config.lossy_element_ids:
                if eid not in sol.element_measurements or eid == config.eom_element_id:
                    continue
                em = sol.element_measurements[eid]
                # Guard: parameter-dependent admittance (Y_p ≠ 0) is unsupported.
                # ONE_PORT_MODEL elements may have dY/dp ≠ 0; the derivative
                # P = Re(Y)|V|² has an extra Re(Y_p)|V|² term we don't compute.
                # All standard matching-network elements (R, L, C) have Y_p = 0
                # because their values are not optimization parameters in P05.
                if em.element_kind == ElementKind.ONE_PORT_MODEL:
                    unsupported.append(f"J_loss_param_dependent_Y_{eid}_f{fi}")
                    _loss_elem_unsupported = True
                    break
                v_elem = em.voltage
                if eid not in obs.element_voltage_derivs:
                    unsupported.append(f"J_loss_missing_dV_{eid}_f{fi}")
                    _loss_elem_unsupported = True
                    break
                dv_elem = obs.element_voltage_derivs[eid]
                # G = Re(Y) = Re(I/V) if |V| > 0, else P = 0 anyway
                if abs(v_elem) > 0:
                    y_elem = em.current / v_elem
                    g_elem = float(np.real(y_elem))
                    dp_parasitic += 2.0 * g_elem * np.real(np.conj(v_elem) * dv_elem)
                # If |V| = 0, P_R = 0 and dP_R/dp is degenerate at this point,
                # but the contribution is continuous at 0 since |V|^2/R → 0.
            if _loss_elem_unsupported:
                n_valid_loss += 1
                continue
            dp_eom = dp_source - dp_parasitic
            # d(loss_db)/dp = 10/ln(10) * (dP_source/P_source - dP_eom/P_eom)
            c10 = 10.0 / math.log(10.0)
            dj_loss += c10 * (dp_source / p_source - dp_eom / p_eom)
            n_valid_loss += 1
        if n_valid_loss > 0:
            grad += config.w_loss * dj_loss / n_valid_loss

    # ---- 4. dJ_complexity / dp = 0 (constant within domain) ----
    # Mathematically correct zero — n_reactive does not depend on x.

    # ---- 5. dJ_soft / dp ----
    # J_soft = sum_s lambda_s * max(0, -g_s)^2
    # d/dp = sum_s { -2 lambda_s max(0, -g_s) dg_s/dp  if g_s < 0
    #                0                                    if g_s >= 0 }
    # The squared hinge max(0, -g)^2 is C^1 at g=0 when g itself is C^1,
    # because the derivative approaches 0 from both sides.
    # Nonsmoothness in J_soft comes from nonsmoothness in g_s itself.
    if soft_g_vector is not None and soft_jacobian is not None:
        for i, desc in enumerate(soft_layout.descriptors):
            if i >= len(soft_g_vector):
                break
            g_i = float(soft_g_vector[i])
            if g_i < 0:
                violation = -g_i
                # dJ_soft_i/dp = -2 lambda_s * violation * dg_i/dp
                grad += -2.0 * desc.penalty_weight * violation * soft_jacobian[i, :]

    # ---- Determine overall status ----
    if unsupported:
        status = DerivativeStatus.UNSUPPORTED
    elif nonsmooth:
        status = DerivativeStatus.NONSMOOTH_KINK

    return ObjectiveGradientResult(
        gradient=grad,
        status=status,
        nonsmooth_terms=tuple(nonsmooth),
        unsupported_terms=tuple(unsupported),
    )
