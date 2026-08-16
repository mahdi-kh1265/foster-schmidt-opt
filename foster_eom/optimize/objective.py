"""Objective function builders for Prompt 05.

Implements:
    J_gamma    — reflection/match penalty
    J_voltage  — EOM voltage tracking penalty
    J_loss     — parasitic loss fraction
    J_complex  — topology complexity (constant within domain)
    J_base     = w_gamma * J_gamma + w_voltage * J_voltage + ...
    J_soft     = sum_s lambda_s * max(0, -g_s)^2
    J_total    = J_base + J_soft    ← single canonical optimizer scalar
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foster_eom.circuit.measurements import CircuitSolution
    from foster_eom.optimize.constraints import ConstraintLayout


# ---------------------------------------------------------------------------
# Objective configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights and reference values for the objective function."""

    z_ref_ohm: float
    w_gamma: float = 1.0
    w_voltage: float = 1.0
    w_loss: float = 0.0
    w_complexity: float = 0.0
    #: voltage targets per target index (V); None = no voltage term for that target
    voltage_targets_rms_v: tuple[float | None, ...] = ()
    #: per-target voltage term weight (defaults to 1.0)
    voltage_target_weights: tuple[float, ...] = ()
    #: EOM element ID for loss accounting (or None)
    eom_element_id: str | None = None
    #: matching-network resistor element IDs for parasitic loss (empty = no loss)
    lossy_element_ids: tuple[str, ...] = ()
    #: n_reactive for complexity term (constant per domain)
    n_reactive: int = 0


# ---------------------------------------------------------------------------
# Individual term calculators
# ---------------------------------------------------------------------------


def compute_j_gamma(
    target_solutions: tuple[CircuitSolution, ...],
    z_ref_ohm: float,
) -> float:
    """Mean squared reflection coefficient over target frequencies.

    J_gamma = (1/N) * sum_i |Gamma_i|^2
    """
    if not target_solutions:
        return 0.0
    total = 0.0
    for sol in target_solutions:
        if sol.gamma is None:
            total += 1.0  # worst case
        else:
            total += abs(sol.gamma) ** 2
    return total / len(target_solutions)


def compute_j_voltage(
    target_solutions: tuple[CircuitSolution, ...],
    voltage_targets_rms_v: tuple[float | None, ...],
    voltage_target_weights: tuple[float, ...],
) -> float:
    """Weighted sum of squared relative EOM-voltage deviations.

    J_voltage = sum_i w_i * ((|V_EOM_i| - V_i*) / max(V_i*, 1e-6))^2

    Only targets with ``voltage_targets_rms_v[i] is not None`` contribute.
    """
    total = 0.0
    n = len(target_solutions)
    for i, sol in enumerate(target_solutions):
        v_target = voltage_targets_rms_v[i] if i < len(voltage_targets_rms_v) else None
        if v_target is None:
            continue
        w = voltage_target_weights[i] if i < len(voltage_target_weights) else 1.0
        if sol.v_eom is None:
            total += w * 1.0  # penalty for missing value
        else:
            v_mag = abs(sol.v_eom)
            v_ref = max(v_target, 1e-6)
            total += w * ((v_mag - v_target) / v_ref) ** 2
    return total


def compute_j_loss(
    target_solutions: tuple[CircuitSolution, ...],
    eom_element_id: str | None,
    lossy_element_ids: tuple[str, ...],
) -> float:
    """Parasitic loss (dB, matched scale): 10 * log10(P_source / P_eom).
    
    Averaged over targets.
    """
    if not lossy_element_ids or not target_solutions:
        return 0.0

    total_loss_db = 0.0
    n_valid = 0
    for sol in target_solutions:
        if sol.element_measurements is None or sol.p_source_delivered_w is None:
            continue
        p_source = sol.p_source_delivered_w
        if p_source <= 0:
            continue
        p_parasitic = sum(
            sol.element_measurements[eid].real_power_w
            for eid in lossy_element_ids
            if eid in sol.element_measurements and eid != eom_element_id
        )
        # Power to EOM is what's left
        p_eom = p_source - p_parasitic

        # Guard against zero or negative EOM power to prevent log error
        if p_eom <= 0:
            # If all power is lost, assign a high dB penalty (e.g., 100 dB)
            loss_db = 100.0
        else:
            loss_db = 10.0 * math.log10(p_source / p_eom)

        total_loss_db += max(0.0, loss_db)
        n_valid += 1

    return total_loss_db / n_valid if n_valid > 0 else 0.0


def compute_j_complexity(n_reactive: int, alpha: float) -> float:
    """Complexity penalty: alpha * n_reactive.

    Constant within a domain; retained for cross-domain comparability.
    """
    return alpha * n_reactive


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Full objective term breakdown for one evaluation."""

    j_gamma: float
    j_voltage: float
    j_loss: float
    j_complexity: float
    j_base: float
    j_soft: float
    j_total: float
    soft_terms: dict[str, float]  # per-soft-constraint name → λ * max(0,-g)^2


def compute_objective(
    config: ObjectiveConfig,
    target_solutions: tuple[CircuitSolution, ...],
    soft_layout: ConstraintLayout,
    soft_g_vector: tuple[float, ...],
) -> ObjectiveBreakdown:
    """Compute the full objective breakdown.

    Parameters
    ----------
    soft_g_vector : tuple[float, ...]
        Soft-constraint margin vector (same length and order as
        ``soft_layout.descriptors``).
    """
    j_gamma = compute_j_gamma(target_solutions, config.z_ref_ohm)
    j_voltage = compute_j_voltage(
        target_solutions,
        config.voltage_targets_rms_v,
        config.voltage_target_weights,
    )
    j_loss = compute_j_loss(
        target_solutions,
        config.eom_element_id,
        config.lossy_element_ids,
    )
    j_complexity = compute_j_complexity(config.n_reactive, config.w_complexity)

    j_base = (
        config.w_gamma * j_gamma
        + config.w_voltage * j_voltage
        + config.w_loss * j_loss
        + j_complexity
    )

    # Soft penalty: λ * max(0, -g)^2
    soft_terms: dict[str, float] = {}
    j_soft = 0.0
    for i, desc in enumerate(soft_layout.descriptors):
        g_i = float(soft_g_vector[i]) if i < len(soft_g_vector) else 0.0
        violation = max(0.0, -g_i)
        penalty = desc.penalty_weight * violation ** 2
        soft_terms[desc.name] = penalty
        j_soft += penalty

    j_total = j_base + j_soft

    return ObjectiveBreakdown(
        j_gamma=j_gamma,
        j_voltage=j_voltage,
        j_loss=j_loss,
        j_complexity=j_complexity,
        j_base=j_base,
        j_soft=j_soft,
        j_total=j_total,
        soft_terms=soft_terms,
    )
