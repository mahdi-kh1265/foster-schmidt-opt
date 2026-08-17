"""View models for P06 verification results."""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.analysis.q_factor import ResonanceQMetrics
from foster_eom.analysis.stress import StressSummary


@dataclass(frozen=True)
class QMetricsRow:
    f0_hz: float
    q_3db: float
    z_peak_ohm: float


@dataclass(frozen=True)
class StressRow:
    element: str
    v_peak: float
    i_peak: float
    p_diss_w: float
    freq_hz: float


@dataclass(frozen=True)
class VerifyVM:
    q_metrics: list[QMetricsRow]
    stress: list[StressRow]

    @classmethod
    def from_results(cls, q_res: list[ResonanceQMetrics], stress_res: StressSummary) -> VerifyVM:
        q_rows = [
            QMetricsRow(
                f0_hz=q.f0_hz,
                q_3db=q.q_3db,
                z_peak_ohm=q.z_peak_ohm,
            )
            for q in q_res
        ]

        stress_rows = []
        for element, s in stress_res.element_stresses.items():
            stress_rows.append(
                StressRow(
                    element=element,
                    v_peak=s.v_peak,
                    i_peak=s.i_peak,
                    p_diss_w=s.p_diss_w,
                    freq_hz=s.freq_hz,
                )
            )

        return cls(q_metrics=q_rows, stress=stress_rows)
