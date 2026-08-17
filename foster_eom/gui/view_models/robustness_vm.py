"""View models for P10 robustness results."""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.robustness.result import RobustnessResult


@dataclass(frozen=True)
class YieldSummaryVM:
    evaluable_yield_pct: float
    yield_lower_bound_pct: float
    yield_upper_bound_pct: float

    n_samples: int
    n_pass: int
    n_physical_fail: int
    n_model_unresolved: int
    n_numerical_unresolved: int

    ci_displayed: bool
    ci_lo_pct: float | None
    ci_hi_pct: float | None
    ci_level_pct: float | None

    yield_p06_pct: float | None
    p06_diagnostic_label: str | None


@dataclass(frozen=True)
class SensitivityRow:
    slot: str
    impact: float


@dataclass(frozen=True)
class RobustnessVM:
    summary: YieldSummaryVM
    sensitivity: list[SensitivityRow]

    @classmethod
    def from_result(cls, r: RobustnessResult) -> RobustnessVM:
        ys = r.yield_stats

        ci_displayed = ys.ci_method == "wilson"

        summary = YieldSummaryVM(
            evaluable_yield_pct=ys.yield_evaluable * 100.0,
            yield_lower_bound_pct=ys.yield_lower_bound * 100.0,
            yield_upper_bound_pct=ys.yield_upper_bound * 100.0,
            n_samples=ys.n_samples,
            n_pass=ys.n_pass,
            n_physical_fail=ys.n_physical_fail,
            n_model_unresolved=ys.n_model_unresolved,
            n_numerical_unresolved=ys.n_numerical_unresolved,
            ci_displayed=ci_displayed,
            ci_lo_pct=(ys.ci_lo * 100.0) if ys.ci_lo is not None else None,
            ci_hi_pct=(ys.ci_hi * 100.0) if ys.ci_hi is not None else None,
            ci_level_pct=(ys.ci_level * 100.0) if ys.ci_level is not None else None,
            yield_p06_pct=(ys.yield_p06 * 100.0) if ys.yield_p06 is not None else None,
            p06_diagnostic_label=r.p06_diagnostic_label,
        )

        sensitivity = []
        if r.oat_sensitivity:
            # Sort by absolute impact descending
            for slot, impact in sorted(
                r.oat_sensitivity.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                sensitivity.append(SensitivityRow(slot=slot, impact=impact))

        return cls(summary=summary, sensitivity=sensitivity)
