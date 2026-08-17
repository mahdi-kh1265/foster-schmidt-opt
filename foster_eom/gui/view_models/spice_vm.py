"""View models for SPICE verification."""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.spice.result import SpiceValidationReport


@dataclass(frozen=True)
class ComparisonRow:
    quantity: str
    max_rel_err: float
    max_phase_deg: float
    status: str


@dataclass(frozen=True)
class SpiceVM:
    status_label: str
    fail_reason: str | None
    unsupported_elements: list[str]
    unsupported_reasons: list[str]
    comparisons: list[ComparisonRow]

    @classmethod
    def from_report(cls, r: SpiceValidationReport) -> SpiceVM:
        status_map = {
            "pass": "PASS",
            "warn": "WARN",
            "fail": "FAIL",
            "unsupported": "UNSUPPORTED",
            "solver_unavailable": "SOLVER UNAVAILABLE",
        }
        status_label = status_map.get(r.status, r.status.upper())

        comparisons = []
        for c in r.comparisons:
            status = "PASS"
            if (
                c.max_rel_err > r.thresholds.fail_max_rel_err
                or c.max_phase_deg > r.thresholds.fail_max_phase_deg
            ):
                status = "FAIL"
            elif (
                c.max_rel_err > r.thresholds.pass_max_rel_err
                or c.max_phase_deg > r.thresholds.pass_max_phase_deg
            ):
                status = "WARN"

            comparisons.append(
                ComparisonRow(
                    quantity=c.quantity,
                    max_rel_err=c.max_rel_err,
                    max_phase_deg=c.max_phase_deg,
                    status=status,
                )
            )

        return cls(
            status_label=status_label,
            fail_reason=r.fail_reason,
            unsupported_elements=list(r.unsupported_elements),
            unsupported_reasons=list(r.unsupported_model_reasons),
            comparisons=comparisons,
        )
