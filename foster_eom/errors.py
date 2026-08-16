"""Domain exceptions and structured warning records.

Every warning carries context (code, severity, message, frequency/component
context, recommended action) so the GUI and reports can present actionable
diagnostics without parsing exception strings.

Custom exceptions follow the categories in spec §35.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Warning severity
# ---------------------------------------------------------------------------

class WarningSeverity(enum.Enum):
    """Warning severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Structured warning record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WarningRecord:
    """Structured warning with context for diagnostics and reports.

    Attributes
    ----------
    code : str
        Machine-readable warning code (e.g. ``MODEL_EXTRAPOLATION_BLOCKED``).
    severity : WarningSeverity
        Severity level.
    message : str
        Human-readable description.
    frequency_hz : float | None
        Frequency at which the warning was generated, if applicable.
    component_id : str | None
        Identifier of the component involved, if applicable.
    context : dict[str, Any]
        Additional structured context.
    recommended_action : str
        Suggested user action.
    """

    code: str
    severity: WarningSeverity
    message: str
    frequency_hz: float | None = None
    component_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class FosterEOMError(Exception):
    """Base exception for all foster_eom errors."""


# ---------------------------------------------------------------------------
# Domain / validation exceptions
# ---------------------------------------------------------------------------

class ProjectValidationError(FosterEOMError):
    """Project specification fails validation."""


class SchemaVersionError(FosterEOMError):
    """Unsupported or unrecognized schema version."""


class SchemaMigrationError(FosterEOMError):
    """Error during schema migration."""


# ---------------------------------------------------------------------------
# Model exceptions
# ---------------------------------------------------------------------------

class ModelValidityError(FosterEOMError):
    """Frequency or parameter is outside model validity range."""


class ModelParseError(FosterEOMError):
    """Error parsing a model file (Touchstone, SPICE, CSV, etc.)."""


# ---------------------------------------------------------------------------
# Topology / synthesis exceptions
# ---------------------------------------------------------------------------

class TopologyError(FosterEOMError):
    """Topology is invalid, infeasible, or violates constraints."""


class FosterSynthesisError(FosterEOMError):
    """Foster/Schmidt synthesis failure (passivity, pole, etc.)."""


# ---------------------------------------------------------------------------
# Circuit engine exceptions
# ---------------------------------------------------------------------------

class CircuitSingularityError(FosterEOMError):
    """MNA matrix is singular or ill-conditioned at evaluation frequency."""


# ---------------------------------------------------------------------------
# Optimization exceptions
# ---------------------------------------------------------------------------

class OptimizationError(FosterEOMError):
    """Optimizer failure that is not a candidate-level invalidity."""


# ---------------------------------------------------------------------------
# Catalog exceptions
# ---------------------------------------------------------------------------

class CatalogError(FosterEOMError):
    """Component catalog/library operation error."""


# ---------------------------------------------------------------------------
# SPICE / verification exceptions
# ---------------------------------------------------------------------------

class SpiceVerificationError(FosterEOMError):
    """ngspice execution or cross-check failure."""


# ---------------------------------------------------------------------------
# Infeasibility reasons  (spec §22.1)
# ---------------------------------------------------------------------------

class InfeasibilityReason(enum.Enum):
    """Structured reasons for design infeasibility."""

    NO_VALID_L_ORIENTATION = "NO_VALID_L_ORIENTATION"
    FOSTER_POSITIVITY_FAILURE = "FOSTER_POSITIVITY_FAILURE"
    POLE_INTERVAL_INFEASIBLE = "POLE_INTERVAL_INFEASIBLE"
    COMPONENT_RANGE_INFEASIBLE = "COMPONENT_RANGE_INFEASIBLE"
    MATCH_CONSTRAINT_INFEASIBLE = "MATCH_CONSTRAINT_INFEASIBLE"
    VOLTAGE_TARGET_INFEASIBLE = "VOLTAGE_TARGET_INFEASIBLE"
    STRESS_LIMIT_INFEASIBLE = "STRESS_LIMIT_INFEASIBLE"
    MODEL_VALIDITY_INSUFFICIENT = "MODEL_VALIDITY_INSUFFICIENT"
    CATALOG_NO_PARTS = "CATALOG_NO_PARTS"
    ROBUSTNESS_FAILURE = "ROBUSTNESS_FAILURE"
    SOLVER_NUMERICAL_FAILURE = "SOLVER_NUMERICAL_FAILURE"


# ---------------------------------------------------------------------------
# Circuit solve status  (spec §B.3)
# ---------------------------------------------------------------------------

class CircuitSolveStatus(enum.Enum):
    """Status of a single-frequency circuit solve."""

    OK = "OK"
    SINGULAR_OR_ILL_CONDITIONED = "SINGULAR_OR_ILL_CONDITIONED"
    MODEL_EXTRAPOLATION = "MODEL_EXTRAPOLATION"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
