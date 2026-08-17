"""Post-optimization analysis: adaptive sweep, Q, stress, time-domain (Prompt 06)."""

from foster_eom.analysis.q_factor import (
    QResult,
    QStatus,
    ResonanceQMetrics,
    compute_q_metrics,
)
from foster_eom.analysis.stress import (
    ElementStress,
    StressSummary,
    compute_stress,
)
from foster_eom.analysis.sweep import (
    ResonancePeak,
    SweepResult,
    SweepSpec,
    compute_adaptive_sweep,
)
from foster_eom.analysis.time_reconstruction import (
    ReconstructedSignal,
    TimeDomainResult,
    TonePhase,
    compute_time_domain,
)

__all__ = [
    "ElementStress",
    "QResult",
    "QStatus",
    "ReconstructedSignal",
    "ResonancePeak",
    "ResonanceQMetrics",
    "StressSummary",
    "SweepResult",
    "SweepSpec",
    "TimeDomainResult",
    "TonePhase",
    "compute_adaptive_sweep",
    "compute_q_metrics",
    "compute_stress",
    "compute_time_domain",
]
