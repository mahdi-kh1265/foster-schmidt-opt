"""Prompt 05 unit tests — preflight.py."""

from __future__ import annotations

import pytest

from foster_eom.domain.objectives import LocalMethod, OptimizationSpec
from foster_eom.optimize.preflight import (
    PreflightError,
    PreflightReport,
    PreflightValidator,
    run_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(**kw) -> OptimizationSpec:
    defaults = dict(
        local_method=LocalMethod.TRUST_CONSTR,
        local_fallback_method=LocalMethod.TRUST_CONSTR,
        workers=1,
        feasibility_tolerance=1e-6,
        near_feasibility_tolerance=0.05,
        max_global_evaluations=5000,
    )
    defaults.update(kw)
    return OptimizationSpec(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreflightSLSQP:
    def test_slsqp_primary_raises(self) -> None:
        with pytest.raises(PreflightError, match="SLSQP"):
            run_preflight(_spec(local_method=LocalMethod.SLSQP))

    def test_slsqp_fallback_raises(self) -> None:
        with pytest.raises(PreflightError, match="SLSQP"):
            run_preflight(_spec(
                local_method=LocalMethod.TRUST_CONSTR,
                local_fallback_method=LocalMethod.SLSQP,
            ))

    def test_trust_constr_passes(self) -> None:
        report = run_preflight(_spec())
        assert report.passed


class TestPreflightWorkers:
    def test_auto_string_valid(self) -> None:
        report = run_preflight(_spec(workers="auto"))
        assert report.passed

    def test_positive_int_valid(self) -> None:
        report = run_preflight(_spec(workers=4))
        assert report.passed

    def test_zero_workers_raises(self) -> None:
        with pytest.raises(PreflightError, match="workers"):
            run_preflight(_spec(workers=0))

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(PreflightError, match="workers"):
            run_preflight(_spec(workers="parallel"))

    def test_float_workers_raises(self) -> None:
        """A float workers value is rejected at Pydantic model construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _spec(workers=2.5)


class TestPreflightNearFeasibility:
    def test_near_equals_feasibility_gets_warning(self) -> None:
        report = run_preflight(_spec(
            feasibility_tolerance=0.01,
            near_feasibility_tolerance=0.01,
        ))
        assert report.passed
        assert any(w.code == "NEAR_FEASIBILITY_TOO_SMALL" for w in report.warnings)

    def test_near_below_feasibility_gets_warning(self) -> None:
        report = run_preflight(_spec(
            feasibility_tolerance=0.05,
            near_feasibility_tolerance=0.01,
        ))
        assert any(w.code == "NEAR_FEASIBILITY_TOO_SMALL" for w in report.warnings)


class TestPreflightBudget:
    def test_small_budget_gets_warning(self) -> None:
        report = run_preflight(_spec(max_global_evaluations=100))
        assert any(w.code == "SMALL_BUDGET" for w in report.warnings)

    def test_default_budget_no_warning(self) -> None:
        report = run_preflight(_spec(max_global_evaluations=50_000))
        assert not any(w.code == "SMALL_BUDGET" for w in report.warnings)


class TestPreflightReport:
    def test_report_type(self) -> None:
        report = run_preflight(_spec())
        assert isinstance(report, PreflightReport)
        assert isinstance(report.warnings, tuple)
        assert isinstance(report.errors, tuple)
