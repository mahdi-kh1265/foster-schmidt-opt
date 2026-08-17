"""P10 statistics: yield, confidence intervals, quantiles, hard-constraint freq.

Wilson CI is only computed for iid random sampling (method="random").
LHS and Sobol produce ci_method=None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from foster_eom.robustness.evaluator import SampleResult
    from foster_eom.robustness.sampler import RobustnessSpec


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QuantileReport:
    """Quantile summary for one scalar distribution."""

    source: str
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float

    @classmethod
    def from_array(cls, values: list[float], source: str) -> QuantileReport:
        if not values:
            nan = float("nan")
            return cls(source=source, p01=nan, p05=nan, p50=nan, p95=nan, p99=nan, min=nan, max=nan)
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            nan = float("nan")
            return cls(source=source, p01=nan, p05=nan, p50=nan, p95=nan, p99=nan, min=nan, max=nan)
        return cls(
            source=source,
            p01=float(np.percentile(arr, 1)),
            p05=float(np.percentile(arr, 5)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            min=float(arr.min()),
            max=float(arr.max()),
        )


@dataclass
class YieldStats:
    """Yield statistics with conservative bounds and optional Wilson CI.

    Bounds
    ------
    yield_lower_bound  = n_pass / n_samples
        Conservative: all unresolved treated as failures.
    yield_evaluable    = n_pass / (n_pass + n_physical_fail)
        Among samples where physics was well-defined.
    yield_upper_bound  = (n_pass + n_unresolved) / n_samples
        Optimistic: all unresolved treated as passes.

    Wilson CI (only for method="random"):
        Applied to yield_evaluable using n_evaluable as denominator.
    yield_p06 is only set when p06_diagnostic="all".
    """

    n_samples: int
    n_pass: int
    n_physical_fail: int
    n_model_unresolved: int
    n_numerical_unresolved: int

    yield_evaluable: float
    yield_lower_bound: float
    yield_upper_bound: float

    yield_p06: float | None = None

    ci_method: str | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    ci_level: float = 0.95

    # Hard-constraint failure frequency:
    # constraint_idx_str → fraction of PHYSICAL_FAIL samples where it is violated
    hard_constraint_failure_freq: dict[str, float] = field(default_factory=dict)


@dataclass
class DistributionStats:
    """Empirical distributions across all evaluable (PASS + PHYSICAL_FAIL) samples."""

    objective: QuantileReport
    v_max: QuantileReport
    v_sum: QuantileReport
    hard_margins: dict[str, QuantileReport]  # "hard_0", "hard_1", ...
    resonance_hz_first: QuantileReport | None
    resonance_hz_worst: QuantileReport | None
    source_current_rms_a: QuantileReport | None = None
    eom_voltage_peak_v: QuantileReport | None = None


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------


def wilson_ci(n_success: int, n_total: int, level: float) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lo, hi) confidence bounds.
    Valid only for iid samples.
    """
    if n_total == 0:
        return (0.0, 1.0)
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    p_hat = n_success / n_total
    denom = 1.0 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return lo, hi


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------


def compute_yield_stats(
    samples: list[SampleResult],
    spec: RobustnessSpec,
    p06_all_run: bool = False,
) -> YieldStats:
    """Compute YieldStats from all sample results."""
    from foster_eom.robustness.evaluator import SampleOutcome

    n = len(samples)
    n_pass = sum(1 for s in samples if s.outcome == SampleOutcome.PASS)
    n_phys = sum(1 for s in samples if s.outcome == SampleOutcome.PHYSICAL_FAIL)
    n_model = sum(1 for s in samples if s.outcome == SampleOutcome.MODEL_COVERAGE_UNRESOLVED)
    n_num = sum(1 for s in samples if s.outcome == SampleOutcome.NUMERICAL_UNRESOLVED)
    n_unresolved = n_model + n_num

    # yield_evaluable denominator: only well-defined physics outcomes
    n_evaluable = n_pass + n_phys
    yield_evaluable = n_pass / n_evaluable if n_evaluable > 0 else float("nan")

    # Conservative bounds over all n_samples
    yield_lb = n_pass / n if n > 0 else float("nan")
    yield_ub = (n_pass + n_unresolved) / n if n > 0 else float("nan")

    # Wilson CI — only for iid random
    ci_method = None
    ci_lo = None
    ci_hi = None
    if spec.method == "random" and n_evaluable > 0:
        ci_method = "wilson"
        ci_lo, ci_hi = wilson_ci(n_pass, n_evaluable, spec.ci_level)

    # yield_p06 only when all samples had P06 run
    yield_p06: float | None = None
    if p06_all_run and n > 0:
        n_p06_pass = sum(1 for s in samples if s.verify_passed is True)
        yield_p06 = n_p06_pass / n

    # Hard-constraint failure frequency
    hc_fail_freq = _hard_constraint_failure_freq(samples)

    return YieldStats(
        n_samples=n,
        n_pass=n_pass,
        n_physical_fail=n_phys,
        n_model_unresolved=n_model,
        n_numerical_unresolved=n_num,
        yield_evaluable=yield_evaluable,
        yield_lower_bound=yield_lb,
        yield_upper_bound=yield_ub,
        yield_p06=yield_p06,
        ci_method=ci_method,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        ci_level=spec.ci_level,
        hard_constraint_failure_freq=hc_fail_freq,
    )


def _hard_constraint_failure_freq(
    samples: list[SampleResult],
) -> dict[str, float]:
    """Compute per-constraint failure frequency across PHYSICAL_FAIL samples."""
    from foster_eom.robustness.evaluator import SampleOutcome

    fail_samples = [s for s in samples if s.outcome == SampleOutcome.PHYSICAL_FAIL]
    if not fail_samples:
        return {}

    n_fail = len(fail_samples)
    counts: dict[str, int] = {}

    for s in fail_samples:
        if s.eval_result is None or not s.eval_result.hard_margins:
            continue
        for i, margin in enumerate(s.eval_result.hard_margins):
            if margin < 0.0:
                key = f"hard_{i}"
                counts[key] = counts.get(key, 0) + 1

    return {k: v / n_fail for k, v in counts.items()}


def compute_distributions(
    samples: list[SampleResult],
) -> DistributionStats:
    """Compute empirical distributions over evaluable samples."""
    from foster_eom.robustness.evaluator import SampleOutcome

    evaluable = [
        s for s in samples if s.outcome in (SampleOutcome.PASS, SampleOutcome.PHYSICAL_FAIL)
    ]

    obj_vals = [s.objective_value for s in evaluable if s.objective_value is not None]
    vmax_vals = [s.v_max for s in evaluable if s.v_max is not None]
    vsum_vals = [s.v_sum for s in evaluable if s.v_sum is not None]

    # Hard margins by index
    n_hard = max(
        (len(s.eval_result.hard_margins) for s in evaluable if s.eval_result is not None),
        default=0,
    )
    hard_margins: dict[str, QuantileReport] = {}
    for i in range(n_hard):
        vals = [
            s.eval_result.hard_margins[i]
            for s in evaluable
            if s.eval_result is not None and len(s.eval_result.hard_margins) > i
        ]
        hard_margins[f"hard_{i}"] = QuantileReport.from_array(vals, f"hard_margin_{i}")

    # Resonance statistics (first and worst/maximum resonance frequency)
    all_resonances = [hz for s in evaluable for hz in s.resonance_hz]
    resonance_first: QuantileReport | None = None
    resonance_worst: QuantileReport | None = None
    if all_resonances:
        first_res = [s.resonance_hz[0] for s in evaluable if s.resonance_hz]
        worst_res = [max(s.resonance_hz) for s in evaluable if s.resonance_hz]
        resonance_first = QuantileReport.from_array(first_res, "resonance_hz_first")
        resonance_worst = QuantileReport.from_array(worst_res, "resonance_hz_worst")

    return DistributionStats(
        objective=QuantileReport.from_array(obj_vals, "objective_value"),
        v_max=QuantileReport.from_array(vmax_vals, "v_max"),
        v_sum=QuantileReport.from_array(vsum_vals, "v_sum"),
        hard_margins=hard_margins,
        resonance_hz_first=resonance_first,
        resonance_hz_worst=resonance_worst,
    )
