"""Analytical sensitivity engine (P12.5-D).

Direct and adjoint parameter sensitivities for the MNA circuit model, the P05
objective, and the hard/soft constraint layouts.  ``DerivativeTransaction`` is
the single entry point used by production code; it owns the lifecycle of one
parameter vector's heavy state.
"""

from foster_eom.sensitivities.coverage import (
    DerivativeCoverage,
    DerivativeRoute,
    get_derivative_coverage,
)
from foster_eom.sensitivities.objective_gradient import (
    DerivativeStatus,
    ObjectiveGradientResult,
    check_analytical_support,
)
from foster_eom.sensitivities.transaction import DerivativeTransaction

__all__ = [
    "DerivativeCoverage",
    "DerivativeRoute",
    "DerivativeStatus",
    "DerivativeTransaction",
    "ObjectiveGradientResult",
    "check_analytical_support",
    "get_derivative_coverage",
]
