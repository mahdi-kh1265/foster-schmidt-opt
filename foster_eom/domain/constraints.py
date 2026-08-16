"""Constraint records (spec §6.7).

All constraints use explicit status and severity.  Hard constraints become
solver constraints; soft constraints become penalty terms.  Each constraint
carries enough metadata for the results report to show normalized margins.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class ConstraintSeverity(enum.StrEnum):
    """Whether a constraint is hard (solver) or soft (penalty)."""

    HARD = "hard"
    SOFT = "soft"


class FrequencyScope(enum.StrEnum):
    """Which frequencies a constraint applies to."""

    ALL_TARGETS = "all_targets"
    SPECIFIC = "specific"
    SWEEP = "sweep"
    OFF_TARGET = "off_target"


class ConstraintRecord(BaseModel, frozen=True):
    """Single engineering constraint (spec §6.7).

    Attributes
    ----------
    name : str
        Descriptive name.
    severity : ConstraintSeverity
        Hard or soft.
    limit : float
        Constraint bound value.
    unit : str
        Unit of the limit.
    frequency_scope : FrequencyScope
        Where the constraint applies.
    specific_frequencies_hz : list[float]
        Frequencies for ``SPECIFIC`` scope.
    penalty_weight : float
        Soft penalty weight (ignored for hard).
    validation_only : bool
        If True, evaluated post-optimization only.
    """

    name: str
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    limit: float = 0.0
    unit: str = ""
    frequency_scope: FrequencyScope = FrequencyScope.ALL_TARGETS
    specific_frequencies_hz: list[float] = Field(default_factory=list)
    penalty_weight: float = Field(default=1.0, ge=0.0)
    validation_only: bool = False


class MatchConstraints(BaseModel, frozen=True):
    """Source-side impedance match constraints.

    Attributes
    ----------
    gamma_max : float
        Maximum reflection coefficient magnitude at targets.
    resistance_min_ohm : float
        Minimum real part of Z_in.
    resistance_max_ohm : float
        Maximum real part of Z_in.
    max_abs_reactance_ohm : float
        Maximum |Im(Z_in)|.
    """

    gamma_max: float = Field(default=0.25, ge=0.0, le=1.0)
    resistance_min_ohm: float = Field(default=35.0, gt=0.0)
    resistance_max_ohm: float = Field(default=70.0, gt=0.0)
    max_abs_reactance_ohm: float = Field(default=20.0, ge=0.0)

    def model_post_init(self, __context: object) -> None:
        """Validate resistance window."""
        if self.resistance_min_ohm > self.resistance_max_ohm:
            raise ValueError(
                f"resistance_min_ohm ({self.resistance_min_ohm}) "
                f"must be ≤ resistance_max_ohm ({self.resistance_max_ohm})"
            )


class QBandwidthConstraints(BaseModel, frozen=True):
    """Q and bandwidth constraints (spec §17).

    Attributes
    ----------
    q_reporting_enabled : bool
        Whether Q extraction is active.
    preferred_q_range : tuple[float, float] | None
        Preferred loaded-Q window.
    q_is_hard_constraint : bool
        Whether Q range is a hard constraint.
    min_usable_half_bandwidth_hz : float
        Minimum usable half-bandwidth around each target.
    voltage_fraction_for_bandwidth : float
        Fraction of V_target defining usable bandwidth.
    """

    q_reporting_enabled: bool = True
    preferred_q_range: tuple[float, float] | None = None
    q_is_hard_constraint: bool = False
    min_usable_half_bandwidth_hz: float = Field(default=50.0e3, ge=0.0)
    voltage_fraction_for_bandwidth: float = Field(default=0.90, gt=0.0, le=1.0)


class StressConstraints(BaseModel, frozen=True):
    """Component and source stress limits (spec §18).

    Attributes
    ----------
    source_current_rms_max_a : float
        Maximum RMS source current.
    default_cap_peak_voltage_v : float
        Default peak voltage limit for capacitors.
    default_ind_peak_current_a : float
        Default peak current limit for inductors.
    off_target_eom_peak_rms_v : float
        Maximum EOM RMS voltage at off-target frequencies.
    """

    source_current_rms_max_a: float = Field(default=0.5, gt=0.0)
    default_cap_peak_voltage_v: float = Field(default=100.0, gt=0.0)
    default_ind_peak_current_a: float = Field(default=1.0, gt=0.0)
    off_target_eom_peak_rms_v: float = Field(default=50.0, gt=0.0)


class RobustnessSpec(BaseModel, frozen=True):
    """Tolerance/robustness analysis configuration (spec §24).

    Attributes
    ----------
    enabled : bool
        Whether robustness analysis is active.
    optimization_scenarios : int
        Number of tolerance scenarios during optimization.
    final_monte_carlo_samples : int
        Number of Monte Carlo samples for final yield.
    default_component_tolerance : float
        Default component tolerance fraction (e.g. 0.02 for 2%).
    eom_c0_tolerance : float
        EOM static capacitance tolerance fraction.
    random_seed : int
        Seed for reproducible tolerance analysis.
    """

    enabled: bool = True
    optimization_scenarios: int = Field(default=32, ge=1)
    final_monte_carlo_samples: int = Field(default=2000, ge=10)
    default_component_tolerance: float = Field(default=0.02, ge=0.0, le=1.0)
    eom_c0_tolerance: float = Field(default=0.03, ge=0.0, le=1.0)
    random_seed: int = 20260815
