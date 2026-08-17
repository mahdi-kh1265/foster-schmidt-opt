"""Unit tests for P10 statistics module."""

from __future__ import annotations

import math

import pytest

from foster_eom.robustness.evaluator import SampleOutcome, SampleResult
from foster_eom.robustness.sampler import RobustnessSpec
from foster_eom.robustness.stats import (
    QuantileReport,
    compute_distributions,
    compute_yield_stats,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(
    outcome: SampleOutcome,
    obj: float | None = None,
    v_max: float | None = None,
    hard_margins: tuple[float, ...] | None = None,
) -> SampleResult:
    from unittest.mock import MagicMock

    eval_result = None
    if outcome in (SampleOutcome.PASS, SampleOutcome.PHYSICAL_FAIL):
        eval_result = MagicMock()
        eval_result.objective_value = obj if obj is not None else 0.5
        eval_result.v_max = (
            v_max if v_max is not None else (0.0 if outcome == SampleOutcome.PASS else 0.5)
        )
        eval_result.v_sum = eval_result.v_max
        eval_result.hard_margins = hard_margins if hard_margins is not None else (0.1,)
        eval_result.numerical_status = "ok"

    s = SampleResult(
        sample_idx=0,
        draw={},
        perturb_methods={},
        outcome=outcome,
        eval_result=eval_result,
        objective_value=obj if obj is not None else (0.5 if eval_result else None),
        v_max=v_max if v_max is not None else (0.0 if outcome == SampleOutcome.PASS else 0.5),
        v_sum=v_max if v_max is not None else (0.0 if outcome == SampleOutcome.PASS else 0.5),
        hard_margin_min=min(hard_margins) if hard_margins else None,
    )
    return s


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_no_samples(self) -> None:
        lo, hi = wilson_ci(0, 0, 0.95)
        assert lo == 0.0
        assert hi == 1.0

    def test_perfect_yield(self) -> None:
        lo, hi = wilson_ci(100, 100, 0.95)
        assert hi <= 1.0
        assert lo > 0.95

    def test_zero_yield(self) -> None:
        lo, hi = wilson_ci(0, 100, 0.95)
        assert lo < 1e-10, f"lo={lo} should be effectively 0"
        assert hi < 0.05

    def test_width_500_samples_97pct(self) -> None:
        """For N=500, yield=0.97: Wilson CI width should be < 0.04."""
        lo, hi = wilson_ci(485, 500, 0.95)
        width = hi - lo
        assert width < 0.04, f"CI width {width:.4f} too large"

    def test_bounds_in_01(self) -> None:
        lo, hi = wilson_ci(47, 50, 0.95)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= hi


# ---------------------------------------------------------------------------
# YieldStats
# ---------------------------------------------------------------------------


class TestComputeYieldStats:
    def test_all_pass(self) -> None:
        samples = [_sample(SampleOutcome.PASS) for _ in range(100)]
        spec = RobustnessSpec(n_samples=100, method="random")
        ys = compute_yield_stats(samples, spec)
        assert ys.n_pass == 100
        assert ys.n_physical_fail == 0
        assert ys.yield_evaluable == pytest.approx(1.0)
        assert ys.yield_lower_bound == pytest.approx(1.0)
        assert ys.yield_upper_bound == pytest.approx(1.0)

    def test_all_fail(self) -> None:
        samples = [_sample(SampleOutcome.PHYSICAL_FAIL) for _ in range(50)]
        spec = RobustnessSpec(n_samples=50, method="random")
        ys = compute_yield_stats(samples, spec)
        assert ys.yield_evaluable == pytest.approx(0.0)

    def test_unresolved_in_denominator_bounds(self) -> None:
        """Unresolved samples must NOT be silently excluded from bounds."""
        samples = (
            [_sample(SampleOutcome.PASS)] * 80
            + [_sample(SampleOutcome.PHYSICAL_FAIL)] * 10
            + [_sample(SampleOutcome.NUMERICAL_UNRESOLVED)] * 10
        )
        spec = RobustnessSpec(n_samples=100, method="random")
        ys = compute_yield_stats(samples, spec)
        # yield_lower_bound = 80/100 (all unresolved as failures)
        assert ys.yield_lower_bound == pytest.approx(0.80)
        # yield_evaluable = 80/90 (only well-defined)
        assert ys.yield_evaluable == pytest.approx(80 / 90)
        # yield_upper_bound = (80+10)/100 (all unresolved as passes)
        assert ys.yield_upper_bound == pytest.approx(0.90)
        # bounds bracket evaluable yield
        assert ys.yield_lower_bound <= ys.yield_evaluable <= ys.yield_upper_bound

    def test_ci_only_for_random(self) -> None:
        samples = [_sample(SampleOutcome.PASS)] * 100 + [_sample(SampleOutcome.PHYSICAL_FAIL)] * 10
        spec_random = RobustnessSpec(n_samples=110, method="random")
        spec_lhs = RobustnessSpec(n_samples=110, method="lhs")
        ys_r = compute_yield_stats(samples, spec_random)
        ys_l = compute_yield_stats(samples, spec_lhs)
        assert ys_r.ci_method == "wilson"
        assert ys_r.ci_lo is not None
        assert ys_l.ci_method is None
        assert ys_l.ci_lo is None

    def test_yield_p06_only_when_all_run(self) -> None:
        samples = [_sample(SampleOutcome.PASS)] * 10
        for s in samples:
            s.verify_passed = True
        spec = RobustnessSpec(method="random")
        # Not all run
        ys = compute_yield_stats(samples, spec, p06_all_run=False)
        assert ys.yield_p06 is None
        # All run
        ys2 = compute_yield_stats(samples, spec, p06_all_run=True)
        assert ys2.yield_p06 == pytest.approx(1.0)

    def test_hard_constraint_failure_freq(self) -> None:
        # 5 PHYSICAL_FAIL samples; constraint 0 violated in all 5, constraint 1 in 2
        samples = [
            _sample(SampleOutcome.PHYSICAL_FAIL, hard_margins=(-0.1, -0.2)) for _ in range(3)
        ] + [_sample(SampleOutcome.PHYSICAL_FAIL, hard_margins=(-0.1, 0.1)) for _ in range(2)]
        spec = RobustnessSpec()
        ys = compute_yield_stats(samples, spec)
        assert "hard_0" in ys.hard_constraint_failure_freq
        assert ys.hard_constraint_failure_freq["hard_0"] == pytest.approx(1.0)  # 5/5
        assert ys.hard_constraint_failure_freq.get("hard_1", 0) == pytest.approx(0.6)  # 3/5


# ---------------------------------------------------------------------------
# QuantileReport
# ---------------------------------------------------------------------------


class TestQuantileReport:
    def test_monotone_order(self) -> None:
        import random

        rng = random.Random(0)
        vals = [rng.uniform(0, 1) for _ in range(500)]
        r = QuantileReport.from_array(vals, "test")
        assert r.p01 <= r.p05 <= r.p50 <= r.p95 <= r.p99

    def test_empty_array_returns_nan(self) -> None:
        r = QuantileReport.from_array([], "test")
        assert math.isnan(r.p50)

    def test_known_values(self) -> None:
        vals = list(range(101))  # 0 to 100
        r = QuantileReport.from_array(vals, "test")
        assert r.p50 == pytest.approx(50.0, abs=1.0)
        assert r.min == 0.0
        assert r.max == 100.0


# ---------------------------------------------------------------------------
# DistributionStats
# ---------------------------------------------------------------------------


class TestComputeDistributions:
    def test_excludes_unresolved(self) -> None:
        """Only PASS and PHYSICAL_FAIL samples contribute to distributions."""
        samples = (
            [_sample(SampleOutcome.PASS, obj=0.5, v_max=0.0)] * 80
            + [_sample(SampleOutcome.PHYSICAL_FAIL, obj=0.9, v_max=0.3)] * 10
            + [_sample(SampleOutcome.NUMERICAL_UNRESOLVED)] * 10
        )
        dist = compute_distributions(samples)
        # 90 evaluable samples; max objective from PHYSICAL_FAIL
        assert dist.objective.max == pytest.approx(0.9)
        assert dist.v_max.max == pytest.approx(0.3)
