"""P11 SPICE export and MNA cross-validation."""

from foster_eom.spice.api import validate_against_mna
from foster_eom.spice.netlist import SpiceNetlist, build_netlist
from foster_eom.spice.ngspice import (
    NgspiceNotFoundError,
    NgspiceRunError,
    detect_ngspice,
    run_ngspice,
)
from foster_eom.spice.result import (
    MeasurementPlan,
    QuantityComparison,
    SpiceValidationReport,
    ValidationThresholds,
)

__all__ = [
    "MeasurementPlan",
    "NgspiceNotFoundError",
    "NgspiceRunError",
    "QuantityComparison",
    "SpiceNetlist",
    "SpiceValidationReport",
    "ValidationThresholds",
    "build_netlist",
    "detect_ngspice",
    "run_ngspice",
    "validate_against_mna",
]
