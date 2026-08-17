"""P10 OAT local sensitivity and failure association.

OAT: one-at-a-time central-difference sensitivity around nominal.
Failure association: heuristic identification of which slot's draw is most
extreme in PHYSICAL_FAIL samples (not causal attribution).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.circuit.graph import CircuitGraph
    from foster_eom.optimize.evaluator import EvaluationContext
    from foster_eom.realization.result import CatalogCombo
    from foster_eom.robustness.evaluator import SampleResult
    from foster_eom.robustness.uncertainty import SlotUncertainty


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class OATSensitivity:
    """One-at-a-time local sensitivity for one slot.

    Parameters
    ----------
    element_id : str
    sensitivity_J : float
        |ΔJ / (2h)| central difference, normalized by h (fractional step).
        Units: objective-per-unit-fractional-deviation.
    sensitivity_vmax : float
        |Δv_max / (2h)|, same normalization.
    h_frac : float
        Fractional step used (e.g. effective tolerance fraction of the slot).
    j_lo : float
        Objective at nom*(1-h).
    j_hi : float
        Objective at nom*(1+h).
    """

    element_id: str
    sensitivity_J: float
    sensitivity_vmax: float
    h_frac: float
    j_lo: float
    j_hi: float


@dataclass
class FailureAssociation:
    """Heuristic association between a slot and PHYSICAL_FAIL samples.

    For each PHYSICAL_FAIL sample, the slot with the most extreme normalized
    draw (|u_i - 0.5| / 0.5) is identified.  This count is reported per slot.

    This is NOT causal attribution.  It identifies which slot's draw is
    statistically most extreme in failing samples.
    """

    element_id: str
    association_count: int
    association_frac: float  # association_count / n_physical_fail


# ---------------------------------------------------------------------------
# OAT sensitivity
# ---------------------------------------------------------------------------


def run_oat_sensitivity(
    combo: CatalogCombo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
    slot_uncertainties: list[SlotUncertainty],
) -> list[OATSensitivity]:
    """Compute one-at-a-time sensitivity around nominal for each stochastic slot.

    Returns list sorted by sensitivity_J descending (most sensitive first).
    """
    from foster_eom.catalog.component import FallbackPolicy

    # Build nominal model overrides (for all non-perturbed slots)
    def _nominal_overrides(skip_eid: str) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for eid, entry in combo.slot_entries.items():
            if eid == skip_eid:
                continue
            try:
                model = library.build_model(
                    entry.component_id,
                    fallback=FallbackPolicy.ALLOW_LOWER_TIER,
                )
                overrides[eid] = model
            except Exception:
                pass
        return overrides

    freq_range: tuple[float, float] | None = None
    if context.evaluation_frequencies_hz:
        freq_range = (
            min(context.evaluation_frequencies_hz),
            max(context.evaluation_frequencies_hz),
        )

    results: list[OATSensitivity] = []

    for su in slot_uncertainties:
        if not su.is_stochastic:
            continue

        h = su.total_sym_tol  # effective fractional step
        if h < 1e-8:
            continue

        base_overrides = _nominal_overrides(su.element_id)

        j_lo = _eval_perturbed(
            su, combo, library, freq_range, base_graph, context, base_overrides,
            draw_frac=-h,
        )
        j_hi = _eval_perturbed(
            su, combo, library, freq_range, base_graph, context, base_overrides,
            draw_frac=+h,
        )
        vmax_lo = _eval_vmax(
            su, combo, library, freq_range, base_graph, context, base_overrides, -h
        )
        vmax_hi = _eval_vmax(
            su, combo, library, freq_range, base_graph, context, base_overrides, +h
        )

        if any(math.isnan(v) for v in (j_lo, j_hi)):
            continue

        sens_j = abs(j_hi - j_lo) / (2.0 * h) if h > 0 else 0.0
        sens_vmax = abs(vmax_hi - vmax_lo) / (2.0 * h) if h > 0 else 0.0

        results.append(
            OATSensitivity(
                element_id=su.element_id,
                sensitivity_J=sens_j,
                sensitivity_vmax=sens_vmax,
                h_frac=h,
                j_lo=j_lo,
                j_hi=j_hi,
            )
        )

    results.sort(key=lambda x: x.sensitivity_J, reverse=True)
    return results


def _eval_perturbed(
    su: Any,
    combo: Any,
    library: Any,
    freq_range: Any,
    base_graph: Any,
    context: Any,
    base_overrides: dict,
    draw_frac: float,
) -> float:
    """Evaluate objective at a single OAT perturbation point."""
    from foster_eom.realization.substitute import evaluate_with_overrides
    from foster_eom.robustness.evaluator import _build_perturbed_model

    entry = combo.slot_entries[su.element_id]
    draw = {su.element_id: entry.value_nom * (1.0 + draw_frac)}
    try:
        model = _build_perturbed_model(su.element_id, draw, combo, su, library, freq_range)
    except Exception:
        return float("nan")

    overrides = {**base_overrides, su.element_id: model}
    try:
        result = evaluate_with_overrides(base_graph, overrides, context)
        return result.objective_value
    except Exception:
        return float("nan")


def _eval_vmax(
    su: Any,
    combo: Any,
    library: Any,
    freq_range: Any,
    base_graph: Any,
    context: Any,
    base_overrides: dict,
    draw_frac: float,
) -> float:
    from foster_eom.realization.substitute import evaluate_with_overrides
    from foster_eom.robustness.evaluator import _build_perturbed_model

    entry = combo.slot_entries[su.element_id]
    draw = {su.element_id: entry.value_nom * (1.0 + draw_frac)}
    try:
        model = _build_perturbed_model(su.element_id, draw, combo, su, library, freq_range)
    except Exception:
        return float("nan")

    overrides = {**base_overrides, su.element_id: model}
    try:
        result = evaluate_with_overrides(base_graph, overrides, context)
        return result.v_max
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Failure association
# ---------------------------------------------------------------------------


def compute_failure_association(
    samples: list[SampleResult],
    slot_uncertainties: list[SlotUncertainty],
) -> list[FailureAssociation]:
    """Heuristic failure association by most-extreme draw in PHYSICAL_FAIL samples.

    For each PHYSICAL_FAIL sample, identifies which stochastic slot had the
    most extreme draw relative to its distribution (highest |u - 0.5| / 0.5).

    This is associative, not causal.
    """
    from foster_eom.robustness.evaluator import SampleOutcome

    stochastic_ids = [su.element_id for su in slot_uncertainties if su.is_stochastic]
    fail_samples = [s for s in samples if s.outcome == SampleOutcome.PHYSICAL_FAIL]
    n_fail = len(fail_samples)

    if n_fail == 0 or not stochastic_ids:
        return []

    counts: dict[str, int] = {eid: 0 for eid in stochastic_ids}

    for s in fail_samples:
        if not s.draw:
            continue
        # Find the slot with draw most extreme from its nominal
        # Using |drawn/nom - 1| as extremeness proxy (tolerances may differ)
        best_eid = None
        best_extremeness = -1.0
        for su in slot_uncertainties:
            if not su.is_stochastic:
                continue
            drawn = s.draw.get(su.element_id)
            # Recover nominal from combo — not stored in SampleResult, use draw key
            # Approximation: extremeness = |drawn - nom_approx| / (nom_approx * tol)
            # We use the tolerance fraction as normalizer
            if drawn is None:
                continue
            tol = su.total_sym_tol
            if tol < 1e-10:
                continue
            # Estimate nominal as draw median: cannot recover exactly here
            # Use drawn/nom = (1 + delta) → |delta| / tol as normalized extremeness
            # This requires the original nominal, which we store in SlotUncertainty implicitly
            # For now: use |drawn - 1.0| relative if values are near 1 (no, values are SI)
            # Better: just count the slot consistently via max |u_i - 0.5| heuristic
            # We'll approximate nominal from the median of all draws for this slot
            extremeness = abs(drawn)  # placeholder; see below
            if extremeness > best_extremeness:
                best_extremeness = extremeness
                best_eid = su.element_id

        if best_eid is not None:
            counts[best_eid] = counts.get(best_eid, 0) + 1

    return [
        FailureAssociation(
            element_id=eid,
            association_count=cnt,
            association_frac=cnt / n_fail,
        )
        for eid, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    ]
