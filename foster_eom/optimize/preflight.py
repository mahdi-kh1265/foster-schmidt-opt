"""Preflight validation for Prompt 05 optimization (Prompt 05).

Validates the optimization configuration before DE begins.  Errors are hard
failures (raise ``PreflightError``).  Warnings are collected and reported
in ``PreflightReport``.
"""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.domain.objectives import LocalMethod, OptimizationSpec


class PreflightError(Exception):
    """Raised when the optimizer configuration is fundamentally invalid."""


@dataclass(frozen=True)
class PreflightWarning:
    """A non-fatal warning from preflight validation."""

    code: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    """Result of preflight validation.

    Attributes
    ----------
    passed : bool
        False only when a ``PreflightError`` was raised.
    errors : tuple[str, ...]
        Collected error messages (usually empty; errors raise instead).
    warnings : tuple[PreflightWarning, ...]
        Non-fatal warnings.
    """

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[PreflightWarning, ...]


class PreflightValidator:
    """Validates the optimization configuration before DE begins.

    Call ``validate()`` to run all checks.  Raises ``PreflightError`` on any
    hard failure.  Returns a ``PreflightReport`` with all collected warnings.
    """

    def __init__(self, opt_spec: OptimizationSpec) -> None:
        self._spec = opt_spec
        self._warnings: list[PreflightWarning] = []

    def validate(self) -> PreflightReport:
        """Run all preflight checks.  Raises ``PreflightError`` on failure."""
        self._check_local_method()
        self._check_workers()
        self._check_near_feasibility()
        self._check_budget_basics()
        return PreflightReport(
            passed=True,
            errors=(),
            warnings=tuple(self._warnings),
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_local_method(self) -> None:
        """SLSQP is not supported in Prompt 05."""
        if self._spec.local_method == LocalMethod.SLSQP:
            raise PreflightError(
                "SLSQP is not supported in Prompt 05 optimization. "
                "Use LocalMethod.TRUST_CONSTR or LocalMethod.IPOPT."
            )
        if self._spec.local_fallback_method == LocalMethod.SLSQP:
            raise PreflightError(
                "SLSQP is not supported as a fallback in Prompt 05 optimization."
            )
        # Check IPOPT availability when explicitly requested
        if self._spec.local_method == LocalMethod.IPOPT:
            try:
                import cyipopt  # type: ignore[import]
            except ImportError:
                self._warnings.append(PreflightWarning(
                    code="IPOPT_UNAVAILABLE",
                    message=(
                        "LocalMethod.IPOPT requested but cyipopt is not installed. "
                        f"Will fall back to {self._spec.local_fallback_method.value}."
                    ),
                ))

    def _check_workers(self) -> None:
        """Validate workers setting."""
        w = self._spec.workers
        if isinstance(w, str):
            if w != "auto":
                raise PreflightError(
                    f"Invalid workers value {w!r}. Must be an int >= 1 or \"auto\"."
                )
        elif isinstance(w, int):
            if w < 1:
                raise PreflightError(
                    f"workers must be >= 1, got {w}."
                )
        else:
            raise PreflightError(
                f"workers must be an int or \"auto\", got {type(w).__name__!r}."
            )

    def _check_near_feasibility(self) -> None:
        """near_feasibility_tolerance must be > feasibility_tolerance."""
        if self._spec.near_feasibility_tolerance <= self._spec.feasibility_tolerance:
            self._warnings.append(PreflightWarning(
                code="NEAR_FEASIBILITY_TOO_SMALL",
                message=(
                    f"near_feasibility_tolerance ({self._spec.near_feasibility_tolerance}) "
                    f"should be > feasibility_tolerance ({self._spec.feasibility_tolerance})."
                ),
            ))

    def _check_budget_basics(self) -> None:
        """Warn if max_global_evaluations is very small."""
        if self._spec.max_global_evaluations < 500:
            self._warnings.append(PreflightWarning(
                code="SMALL_BUDGET",
                message=(
                    f"max_global_evaluations={self._spec.max_global_evaluations} is very small. "
                    "DE may not converge meaningfully."
                ),
            ))


def run_preflight(opt_spec: OptimizationSpec) -> PreflightReport:
    """Convenience wrapper: run all preflight checks and return report."""
    return PreflightValidator(opt_spec).validate()
