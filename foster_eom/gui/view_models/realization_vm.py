"""View models for catalog realization results."""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.realization.result import RealizationResult


@dataclass(frozen=True)
class SlotRow:
    element_id: str
    vendor: str
    part_number: str
    tier: str
    validity: str


@dataclass(frozen=True)
class RealizationVM:
    status_label: str
    degradation_pct: float | None
    slots: list[SlotRow]

    @classmethod
    def from_result(cls, r: RealizationResult) -> RealizationVM:
        status_map = {
            "feasible": "FEASIBLE",
            "degraded": "DEGRADED",
            "infeasible": "INFEASIBLE (exhaustive)",
            "no_feasible_found": "NO FEASIBLE FOUND (beam)",
            "no_candidates": "NO CANDIDATES",
        }
        status_label = status_map.get(r.status, r.status.upper())

        degradation_pct = None
        if r.degradation is not None and r.continuous_baseline.objective_value != 0:
            degradation_pct = (r.degradation / abs(r.continuous_baseline.objective_value)) * 100.0

        slots = []
        if r.best is not None:
            for s in r.best.slot_entries:
                model = s.component.model
                slots.append(
                    SlotRow(
                        element_id=s.element_id,
                        vendor=s.component.vendor,
                        part_number=s.component.part_number,
                        tier=s.component.tier.value,
                        validity=f"{model.validity_hz[0] / 1e6:.1f}-{model.validity_hz[1] / 1e6:.1f} MHz"
                        if model.validity_hz
                        else "All",
                    )
                )

        return cls(status_label=status_label, degradation_pct=degradation_pct, slots=slots)
