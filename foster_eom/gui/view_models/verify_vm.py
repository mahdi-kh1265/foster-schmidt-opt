"""View models for P06 verification results."""

from __future__ import annotations

from dataclasses import dataclass

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
    def from_results(cls, q_res: object, stress_res: StressSummary) -> VerifyVM:
        from foster_eom.analysis.q_factor import QResult

        q_rows = []
        if isinstance(q_res, QResult):
            for q in q_res.per_target:
                q_rows.append(
                    QMetricsRow(
                        f0_hz=q.f0_hz if q.f0_hz is not None else float("nan"),
                        q_3db=q.q_voltage if q.q_voltage is not None else float("nan"),
                        z_peak_ohm=float("nan"),  # Not available in current ResonanceQMetrics
                    )
                )

        stress_rows = []
        for s in stress_res.elements:
            stress_rows.append(
                StressRow(
                    element=s.element_id,
                    v_peak=max(s.sweep_v_peak_v, s.multitone_v_peak_bound_v),
                    i_peak=max(s.sweep_i_peak_a, s.multitone_i_peak_bound_a),
                    p_diss_w=s.sweep_p_loss_w,
                    freq_hz=s.sweep_worst_v_freq_hz,
                )
            )

        return cls(q_metrics=q_rows, stress=stress_rows)
