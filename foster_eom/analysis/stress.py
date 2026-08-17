"""Per-element frequency-domain stress (Prompt 06, spec ss18).

Two modes
---------
Mode A -- single-tone sweep envelope
    Each verification frequency independently (one tone at a time).
    Peak = sqrt(2) x |V_rms| or |I_rms| at each frequency.
    Purpose: detect dangerous hidden resonances across the full band.

Mode B -- multi-tone operating stress (commanded target tones)
    K simultaneous tones at distinct frequencies:
      V_rms  = sqrt(sum_k |V_k|^2)               (RSS, orthogonal tones)
      I_rms  = sqrt(sum_k |I_k|^2)
      P_avg  = sum_k Re(V_k . conj(I_k))          (sum, not max)
      V_peak_bound = sqrt(2) . sum_k |V_k|         (conservative analytical)
      I_peak_bound = sqrt(2) . sum_k |I_k|

Public API
----------
ElementStress   per-element result
StressSummary   aggregate result
compute_stress(graph, source_spec, sweep_result, target_hz, ...) -> StressSummary
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from foster_eom.analysis.sweep import SweepResult
from foster_eom.circuit.graph import CircuitGraph
from foster_eom.circuit.solve import solve_circuit_single
from foster_eom.domain.source import SourceSpec

_EPS = 1e-30
_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# ElementStress
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementStress:
    """Stress metrics for one circuit element.

    Voltage/current margins: (allowed - actual_peak_bound) / allowed.
    Negative margin = rating violation.  None if rating unknown.
    """

    element_id: str

    # Mode A: sweep envelope
    sweep_v_peak_v: float
    sweep_i_peak_a: float
    sweep_p_loss_w: float
    sweep_worst_v_freq_hz: float
    sweep_worst_i_freq_hz: float
    sweep_worst_p_freq_hz: float

    # Mode B: multi-tone operating
    multitone_v_rms_v: float
    multitone_i_rms_a: float
    multitone_p_avg_w: float
    multitone_v_peak_bound_v: float
    multitone_i_peak_bound_a: float
    multitone_v_peak_reconstructed_v: float | None
    multitone_i_peak_reconstructed_a: float | None

    # Ratings
    rating_voltage_v: float | None
    rating_current_a: float | None
    voltage_derating_factor: float
    current_derating_factor: float
    allowed_voltage_v: float | None
    allowed_current_a: float | None

    # Margins
    sweep_voltage_margin: float | None
    sweep_current_margin: float | None
    multitone_voltage_margin: float | None
    multitone_current_margin: float | None

    stress_complete: bool


# ---------------------------------------------------------------------------
# StressSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StressSummary:
    """Aggregate stress result across all elements."""

    elements: tuple[ElementStress, ...]
    worst_sweep_voltage_element: str | None
    worst_sweep_current_element: str | None
    worst_multitone_voltage_element: str | None
    worst_multitone_current_element: str | None
    verification_complete: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_margin(allowed: float | None, actual: float) -> float | None:
    if allowed is None or allowed <= _EPS:
        return None
    return (allowed - actual) / allowed


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_stress(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    sweep_result: SweepResult,
    target_hz: Sequence[float],
    ratings: dict[str, dict[str, float]] | None = None,
    voltage_derating: float = 1.0,
    current_derating: float = 1.0,
    time_domain_v_peak: dict[str, float] | None = None,
    time_domain_i_peak: dict[str, float] | None = None,
) -> StressSummary:
    """Compute per-element stress in both modes.

    Parameters
    ----------
    graph : CircuitGraph
    source_spec : SourceSpec
    sweep_result : SweepResult
        Used for Mode A (sweep envelope).
    target_hz : Sequence[float]
        Commanded tone frequencies for Mode B.
    ratings : dict or None
        {element_id: {"voltage_v": float, "current_a": float}}.
    voltage_derating, current_derating : float
        Derating fractions applied to all ratings.
    time_domain_v_peak, time_domain_i_peak : dict or None
        Reconstructed peak voltages/currents from TimeDomainResult.

    Returns
    -------
    StressSummary
    """
    from foster_eom.errors import CircuitSolveStatus

    ratings = ratings or {}
    element_ids = list(graph.elements.keys())

    # Mode A: iterate over ALL sweep frequencies
    # accumulators: element_id -> list of (v_rms, i_rms, p_loss, f)
    mode_a: dict[str, list[tuple[float, float, float, float]]] = {eid: [] for eid in element_ids}
    for f in sweep_result.frequencies_hz:
        try:
            sol = solve_circuit_single(graph, source_spec, f)
        except Exception:
            continue
        if sol.status != CircuitSolveStatus.OK or sol.element_measurements is None:
            continue
        for eid in element_ids:
            meas = sol.element_measurements.get(eid)
            if meas is None:
                continue
            mode_a[eid].append(
                (abs(meas.voltage), abs(meas.current), max(meas.real_power_w, 0.0), f)
            )

    # Mode B: solve at each target tone
    mode_b_v: dict[str, list[complex]] = {eid: [] for eid in element_ids}
    mode_b_i: dict[str, list[complex]] = {eid: [] for eid in element_ids}
    mode_b_s: dict[str, list[complex]] = {eid: [] for eid in element_ids}

    for f in target_hz:
        try:
            sol = solve_circuit_single(graph, source_spec, f)
        except Exception:
            continue
        if sol.status != CircuitSolveStatus.OK or sol.element_measurements is None:
            continue
        for eid in element_ids:
            meas = sol.element_measurements.get(eid)
            if meas is None:
                continue
            mode_b_v[eid].append(meas.voltage)
            mode_b_i[eid].append(meas.current)
            mode_b_s[eid].append(meas.complex_power)

    # Assemble ElementStress
    element_results: list[ElementStress] = []
    verification_complete = True

    for eid in element_ids:
        a_data = mode_a[eid]
        if a_data:
            sweep_v_peak = max(_SQRT2 * v for v, _, _, _ in a_data)
            sweep_i_peak = max(_SQRT2 * i for _, i, _, _ in a_data)
            sweep_p_loss = max(p for _, _, p, _ in a_data)
            worst_v_f = max(a_data, key=lambda x: x[0])[3]
            worst_i_f = max(a_data, key=lambda x: x[1])[3]
            worst_p_f = max(a_data, key=lambda x: x[2])[3]
        else:
            sweep_v_peak = sweep_i_peak = sweep_p_loss = 0.0
            worst_v_f = worst_i_f = worst_p_f = 0.0

        v_phasors = mode_b_v[eid]
        i_phasors = mode_b_i[eid]
        s_phasors = mode_b_s[eid]

        if v_phasors:
            mt_v_rms = float((sum(abs(v) ** 2 for v in v_phasors)) ** 0.5)
            mt_i_rms = float((sum(abs(i) ** 2 for i in i_phasors)) ** 0.5)
            mt_p_avg = float(sum(s.real for s in s_phasors))
            mt_v_peak_bound = _SQRT2 * sum(abs(v) for v in v_phasors)
            mt_i_peak_bound = _SQRT2 * sum(abs(i) for i in i_phasors)
        else:
            mt_v_rms = mt_i_rms = mt_p_avg = 0.0
            mt_v_peak_bound = mt_i_peak_bound = 0.0

        rat = ratings.get(eid, {})
        rat_v = rat.get("voltage_v")
        rat_i = rat.get("current_a")
        allowed_v = (rat_v * voltage_derating) if rat_v is not None else None
        allowed_i = (rat_i * current_derating) if rat_i is not None else None

        stress_complete = rat_v is not None and rat_i is not None
        if not stress_complete:
            verification_complete = False

        td_v = time_domain_v_peak.get(eid) if time_domain_v_peak else None
        td_i = time_domain_i_peak.get(eid) if time_domain_i_peak else None

        element_results.append(
            ElementStress(
                element_id=eid,
                sweep_v_peak_v=sweep_v_peak,
                sweep_i_peak_a=sweep_i_peak,
                sweep_p_loss_w=sweep_p_loss,
                sweep_worst_v_freq_hz=worst_v_f,
                sweep_worst_i_freq_hz=worst_i_f,
                sweep_worst_p_freq_hz=worst_p_f,
                multitone_v_rms_v=mt_v_rms,
                multitone_i_rms_a=mt_i_rms,
                multitone_p_avg_w=mt_p_avg,
                multitone_v_peak_bound_v=mt_v_peak_bound,
                multitone_i_peak_bound_a=mt_i_peak_bound,
                multitone_v_peak_reconstructed_v=td_v,
                multitone_i_peak_reconstructed_a=td_i,
                rating_voltage_v=rat_v,
                rating_current_a=rat_i,
                voltage_derating_factor=voltage_derating,
                current_derating_factor=current_derating,
                allowed_voltage_v=allowed_v,
                allowed_current_a=allowed_i,
                sweep_voltage_margin=_safe_margin(allowed_v, sweep_v_peak),
                sweep_current_margin=_safe_margin(allowed_i, sweep_i_peak),
                multitone_voltage_margin=_safe_margin(allowed_v, mt_v_peak_bound),
                multitone_current_margin=_safe_margin(allowed_i, mt_i_peak_bound),
                stress_complete=stress_complete,
            )
        )

    from collections.abc import Callable
    from typing import Any

    def _worst_eid(key_fn: Callable[[Any], Any]) -> str | None:
        if not element_results:
            return None
        return max(element_results, key=key_fn).element_id

    return StressSummary(
        elements=tuple(element_results),
        worst_sweep_voltage_element=_worst_eid(lambda e: e.sweep_v_peak_v),
        worst_sweep_current_element=_worst_eid(lambda e: e.sweep_i_peak_a),
        worst_multitone_voltage_element=_worst_eid(lambda e: e.multitone_v_peak_bound_v),
        worst_multitone_current_element=_worst_eid(lambda e: e.multitone_i_peak_bound_a),
        verification_complete=verification_complete,
    )
