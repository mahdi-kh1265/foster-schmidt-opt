"""Per-slot neighborhood generation from the P08 catalog (Prompt 09).

``build_slot_specs()``   — auto-build SlotSpec list from EvaluationContext + domain.
``build_neighborhoods()``— query catalog and return per-slot NeighborhoodEntry lists.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from foster_eom.catalog.component import ComponentKind
from foster_eom.catalog.query import ComponentQuery
from foster_eom.optimize.variable_map import BranchCoordinates
from foster_eom.realization.spec import NeighborhoodEntry, SlotSpec

if TYPE_CHECKING:
    from foster_eom.catalog.component import ModelCondition
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.optimize.evaluator import EvaluationContext


# ---------------------------------------------------------------------------
# Slot auto-builder
# ---------------------------------------------------------------------------


def build_slot_specs(
    context: EvaluationContext,
    b1: BranchCoordinates,
    b2: BranchCoordinates,
    *,
    value_ratio: float = 1.5,
    voltage_min_v: float | None = None,
    current_min_a: float | None = None,
    current_sat_min_a: float | None = None,
    fallback_policy_l: str = "strict",
    fallback_policy_c: str = "strict",
    in_stock_only: bool = False,
) -> tuple[SlotSpec, ...]:
    """Auto-build SlotSpec tuples from EvaluationContext + unpacked branch coords.

    The verification band ``(sweep_f_min_hz, sweep_f_max_hz)`` is read from
    ``context.evaluation_frequencies_hz`` (min/max of the full grid).
    """
    from foster_eom.catalog.component import FallbackPolicy
    from foster_eom.foster.schmidt import BranchRealization

    f_min = min(context.evaluation_frequencies_hz)
    f_max = max(context.evaluation_frequencies_hz)
    freq_range: tuple[float, float] = (f_min, f_max)

    fp_l = FallbackPolicy(fallback_policy_l)
    fp_c = FallbackPolicy(fallback_policy_c)

    specs: list[SlotSpec] = []

    domain = context.domain

    for _branch_idx, (realization, b, prefix) in enumerate(
        [
            (domain.branch1_realization, b1, "b1"),
            (domain.branch2_realization, b2, "b2"),
        ],
        start=1,
    ):
        if realization != BranchRealization.FINITE_FOSTER:
            continue

        # C0 endpoint
        if b.k0 is not None:
            c0_f = 1.0 / b.k0
            specs.append(
                SlotSpec(
                    element_id=f"{prefix}_C0",
                    value_nom=c0_f,
                    value_ratio=value_ratio,
                    freq_range_hz=freq_range,
                    voltage_min_v=voltage_min_v,
                    current_min_a=current_min_a,
                    current_sat_min_a=current_sat_min_a,
                    fallback_policy=fp_c,
                    in_stock_only=in_stock_only,
                )
            )

        # Foster cells: L_m and C_m
        for cell_i, (l_val, c_val) in enumerate(
            zip(b.l_values_h, b.c_values_f, strict=True), start=1
        ):
            if l_val > 0:
                specs.append(
                    SlotSpec(
                        element_id=f"{prefix}_L{cell_i}",
                        value_nom=l_val,
                        value_ratio=value_ratio,
                        freq_range_hz=freq_range,
                        voltage_min_v=voltage_min_v,
                        current_min_a=current_min_a,
                        current_sat_min_a=current_sat_min_a,
                        fallback_policy=fp_l,
                        in_stock_only=in_stock_only,
                    )
                )
            if c_val > 0:
                specs.append(
                    SlotSpec(
                        element_id=f"{prefix}_C{cell_i}",
                        value_nom=c_val,
                        value_ratio=value_ratio,
                        freq_range_hz=freq_range,
                        voltage_min_v=voltage_min_v,
                        current_min_a=current_min_a,
                        current_sat_min_a=current_sat_min_a,
                        fallback_policy=fp_c,
                        in_stock_only=in_stock_only,
                    )
                )

        # L_inf endpoint
        if b.k_inf is not None:
            l_inf = b.k_inf
            specs.append(
                SlotSpec(
                    element_id=f"{prefix}_Linf",
                    value_nom=l_inf,
                    value_ratio=value_ratio,
                    freq_range_hz=freq_range,
                    voltage_min_v=voltage_min_v,
                    current_min_a=current_min_a,
                    current_sat_min_a=current_sat_min_a,
                    fallback_policy=fp_l,
                    in_stock_only=in_stock_only,
                )
            )

    return tuple(specs)


# ---------------------------------------------------------------------------
# Neighborhood builder
# ---------------------------------------------------------------------------

_SLOT_ID_TO_KIND: dict[str, ComponentKind] = {}


def _infer_kind(element_id: str) -> ComponentKind:
    """Infer L or C from element_id suffix."""
    # element_id e.g. "b1_C0", "b1_L1", "b2_Linf", "b2_C2"
    stem = element_id.upper()
    if "_C" in stem or stem.endswith("_C"):
        return ComponentKind.CAPACITOR
    if "_L" in stem:
        return ComponentKind.INDUCTOR
    raise ValueError(f"Cannot infer component kind from element_id {element_id!r}")


def build_neighborhoods(
    slot_specs: tuple[SlotSpec, ...],
    library: ComponentLibrary,
    k_max: int = 5,
) -> dict[str, list[NeighborhoodEntry]]:
    """Query the catalog and return per-slot NeighborhoodEntry lists.

    Each entry binds a specific ``(component_id, model_condition_id)``
    resolved at this point so later catalog changes cannot affect the result.

    Returns
    -------
    dict[str, list[NeighborhoodEntry]]
        Maps ``slot_spec.element_id`` → list of entries sorted by log_ratio
        (closest first), truncated to k_max.
        Empty list means no catalog parts found for that slot.
    """
    neighborhoods: dict[str, list[NeighborhoodEntry]] = {}

    for slot in slot_specs:
        kind = _infer_kind(slot.element_id)
        v = slot.value_nom
        r = slot.value_ratio

        # Note: freq_range_hz is NOT passed to the SQL query because that
        # filter requires validity_hz_lo IS NOT NULL, which excludes ideal
        # models (where NULL means "valid over all frequencies").
        # Frequency coverage is checked per-MC in _select_best_mc instead.
        q = ComponentQuery(
            kind=kind,
            value_min=v / r,
            value_max=v * r,
            package=slot.package,
            voltage_min_v=slot.voltage_min_v,
            current_min_a=slot.current_min_a,
            current_sat_min_a=slot.current_sat_min_a,
            model_tier_min=slot.required_tier,
            in_stock_only=slot.in_stock_only,
        )
        components = library.query(q)

        entries: list[NeighborhoodEntry] = []
        for comp in components:
            # Resolve the best model_condition_id for this component
            conditions = library.get_model_conditions(comp.id)
            if not conditions:
                continue  # no model at all — skip

            # Pick highest-tier valid condition (respecting freq_range)
            selected_mc = _select_best_mc(conditions, slot)
            if selected_mc is None:
                continue  # STRICT: no valid condition at required tier

            log_r = abs(math.log(comp.value_nom / v)) if comp.value_nom > 0 and v > 0 else math.inf
            entries.append(
                NeighborhoodEntry(
                    component_id=comp.id,
                    model_condition_id=selected_mc.id,
                    vendor=comp.vendor,
                    part_number=comp.part_number,
                    value_nom=comp.value_nom,
                    value_tol_frac=comp.value_tol_frac,
                    model_tier=selected_mc.model_tier,
                    log_ratio=log_r,
                )
            )

        # Sort by log_ratio (closest first), truncate to k_max
        entries.sort(key=lambda e: e.log_ratio)
        neighborhoods[slot.element_id] = entries[:k_max]

    return neighborhoods


def _select_best_mc(
    conditions: list[ModelCondition],
    slot: SlotSpec,
) -> ModelCondition | None:
    """Select the best model condition for a slot, respecting STRICT policy.

    Strategy:
    - Sort conditions by tier rank (highest first).
    - Iterate; skip conditions that don't cover the required freq_range.
    - Return the first that passes all checks.
    - STRICT: if a higher-tier condition exists but fails freq coverage,
      we continue trying lower tiers (the policy prevents falling back
      when a *valid* higher-tier model was already found).
    - If required_tier is set and no condition meets it, return None.
    """
    from foster_eom.catalog.component import FallbackPolicy, tier_rank

    # Sort by tier rank descending (highest first)
    sorted_mc = sorted(conditions, key=lambda mc: tier_rank(mc.model_tier), reverse=True)

    for mc in sorted_mc:
        # Check minimum tier requirement
        if slot.required_tier is not None and tier_rank(mc.model_tier) < tier_rank(
            slot.required_tier
        ):
            # All remaining will be lower — nothing viable
            return None

        vr = mc.validity_hz()

        # Check frequency coverage:
        # - vr is None: model has no stored validity range (ideal mathematical model
        #   has no numerical frequency boundary) → accept it
        # - vr is (lo, hi): must fully cover slot.freq_range_hz
        if slot.freq_range_hz is not None and vr is not None:
            lo, hi = slot.freq_range_hz
            if vr[0] <= lo and vr[1] >= hi:
                return mc  # covers the band — accept

            # Does not cover. Check fallback policy.
            if slot.fallback_policy == FallbackPolicy.STRICT:
                # STRICT: if a higher-tier available model does not cover the band,
                # do not silently downgrade. Mark ineligible.
                return None

            # ALLOW_LOWER_TIER: try the next (lower) tier
            continue

        # No freq filter, or no numerical frequency boundary → accept this condition
        return mc

    return None
