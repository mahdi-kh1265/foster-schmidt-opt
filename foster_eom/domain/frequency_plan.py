"""Frequency plan and target definitions (spec §6.3).

Target frequencies must be unique, positive, and stored in ascending order.
Each target carries optional voltage/modulation requirements and per-target
constraint overrides.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class FrequencyTarget(BaseModel, frozen=True):
    """Single target frequency with associated requirements.

    Attributes
    ----------
    label : str
        Human-readable label (e.g. ``"f1"``).
    frequency_hz : float
        Target frequency in Hz (must be > 0).
    enabled : bool
        Whether this target participates in optimization.
    voltage_target_rms_v : float | None
        Exact desired EOM RMS voltage (V).
    voltage_min_rms_v : float | None
        Minimum acceptable EOM RMS voltage (V).
    voltage_max_rms_v : float | None
        Maximum acceptable EOM RMS voltage (V).
    required_beta : float | None
        Future: optical modulation index target.
    required_phase_deg : float | None
        Optional required source phase at this target (degrees).
    voltage_weight : float
        Relative weight in voltage-error objective (default 1.0).
    match_override : dict | None
        Per-frequency match constraint override.
    notes : str
        Annotation.
    """

    label: str = ""
    frequency_hz: float = Field(gt=0.0)
    enabled: bool = True
    voltage_target_rms_v: float | None = Field(default=None, ge=0.0)
    voltage_min_rms_v: float | None = Field(default=None, ge=0.0)
    voltage_max_rms_v: float | None = Field(default=None, ge=0.0)
    required_beta: float | None = None
    required_phase_deg: float | None = None
    voltage_weight: float = Field(default=1.0, ge=0.0)
    match_override: dict | None = None  # type: ignore[type-arg]
    notes: str = ""

    @model_validator(mode="after")
    def _validate_voltage_range(self) -> FrequencyTarget:
        """Ensure min ≤ target ≤ max when all are specified."""
        vmin = self.voltage_min_rms_v
        vmax = self.voltage_max_rms_v
        vtgt = self.voltage_target_rms_v

        if vmin is not None and vmax is not None and vmin > vmax:
            raise ValueError(
                f"voltage_min_rms_v ({vmin}) must be ≤ voltage_max_rms_v ({vmax})"
            )
        if vtgt is not None:
            if vmin is not None and vtgt < vmin:
                raise ValueError(
                    f"voltage_target_rms_v ({vtgt}) must be ≥ voltage_min_rms_v ({vmin})"
                )
            if vmax is not None and vtgt > vmax:
                raise ValueError(
                    f"voltage_target_rms_v ({vtgt}) must be ≤ voltage_max_rms_v ({vmax})"
                )
        return self


class ExclusionBand(BaseModel, frozen=True):
    """Frequency band excluded from pole placement or target use."""

    f_min_hz: float = Field(gt=0.0)
    f_max_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _validate_band(self) -> ExclusionBand:
        if self.f_min_hz >= self.f_max_hz:
            raise ValueError(
                f"Exclusion band f_min ({self.f_min_hz}) must be < f_max ({self.f_max_hz})"
            )
        return self


class FrequencyPlan(BaseModel, frozen=True):
    """Complete frequency plan (spec §6.3).

    Attributes
    ----------
    targets : list[FrequencyTarget]
        Ordered list of target frequencies.
    sweep_f_min_hz : float
        Lower bound of the verification/sweep band.
    sweep_f_max_hz : float
        Upper bound of the verification/sweep band.
    base_grid_points : int
        Number of base sweep points.
    adaptive_sweep_enabled : bool
        Whether adaptive refinement is active.
    adaptive_peak_tol : float
        Tolerance for adaptive peak detection.
    exclusion_bands : list[ExclusionBand]
        Bands excluded from pole placement.
    """

    targets: list[FrequencyTarget] = Field(min_length=1)
    sweep_f_min_hz: float = Field(gt=0.0)
    sweep_f_max_hz: float = Field(gt=0.0)
    base_grid_points: int = Field(default=1201, ge=10)
    adaptive_sweep_enabled: bool = True
    adaptive_peak_tol: float = Field(default=0.01, gt=0.0)
    exclusion_bands: list[ExclusionBand] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_plan(self) -> FrequencyPlan:
        # Sweep band ordering
        if self.sweep_f_min_hz >= self.sweep_f_max_hz:
            raise ValueError(
                f"sweep_f_min_hz ({self.sweep_f_min_hz}) must be < "
                f"sweep_f_max_hz ({self.sweep_f_max_hz})"
            )

        # Collect enabled target frequencies
        enabled = [t for t in self.targets if t.enabled]
        freqs = [t.frequency_hz for t in enabled]

        # Check for duplicates (within deduplication tolerance of 1 Hz)
        sorted_f = sorted(freqs)
        for i in range(1, len(sorted_f)):
            if abs(sorted_f[i] - sorted_f[i - 1]) < 1.0:
                raise ValueError(
                    f"Duplicate or near-duplicate target frequencies: "
                    f"{sorted_f[i - 1]} Hz and {sorted_f[i]} Hz"
                )

        # Targets should be within sweep band (warning-level, not hard block,
        # but we enforce it here for schema validity)
        for f in freqs:
            if f < self.sweep_f_min_hz or f > self.sweep_f_max_hz:
                raise ValueError(
                    f"Target frequency {f} Hz is outside verification band "
                    f"[{self.sweep_f_min_hz}, {self.sweep_f_max_hz}]"
                )

        return self

    @property
    def enabled_targets(self) -> list[FrequencyTarget]:
        """Return only enabled targets, sorted by frequency."""
        return sorted(
            [t for t in self.targets if t.enabled],
            key=lambda t: t.frequency_hz,
        )

    @property
    def target_frequencies_hz(self) -> list[float]:
        """Sorted list of enabled target frequencies in Hz."""
        return [t.frequency_hz for t in self.enabled_targets]
