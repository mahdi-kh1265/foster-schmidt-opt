"""Topology search and pole specification (spec §6.4, §6.5).

Defines the allowed L-network orientations, Foster cell counts, and pole
placement modes/constraints.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field, model_validator


class LOrientation(enum.StrEnum):
    """L-network orientation options."""

    SCHMIDT_SHUNT_THEN_SERIES = "schmidt_shunt_then_series"
    ALTERNATE_L_ORIENTATION = "alternate_l_orientation"


class PoleMode(enum.StrEnum):
    """Pole specification modes (spec §6.5)."""

    FIXED = "fixed"
    INTERVALS = "intervals"
    AUTO = "auto"
    SCHMIDT_SEED = "schmidt_seed"


class PoleInterval(BaseModel, frozen=True):
    """Constraint on a single pole location.

    Attributes
    ----------
    min_hz : float
        Lower bound for pole frequency.
    max_hz : float
        Upper bound for pole frequency.
    initial_hz : float | None
        Optional initial value hint.
    locked : bool
        If True, pole is fixed at initial_hz.
    """

    min_hz: float = Field(gt=0.0)
    max_hz: float = Field(gt=0.0)
    initial_hz: float | None = None
    locked: bool = False

    @model_validator(mode="after")
    def _validate_interval(self) -> PoleInterval:
        if self.min_hz > self.max_hz:
            raise ValueError(
                f"Pole interval min_hz ({self.min_hz}) must be ≤ max_hz ({self.max_hz})"
            )
        if self.initial_hz is not None and (
            self.initial_hz < self.min_hz or self.initial_hz > self.max_hz
        ):
            raise ValueError(
                f"initial_hz ({self.initial_hz}) must be within [{self.min_hz}, {self.max_hz}]"
            )
        if self.locked and self.initial_hz is None:
            raise ValueError("locked pole requires initial_hz")
        return self


class PoleSpec(BaseModel, frozen=True):
    """Pole placement specification (spec §6.5).

    Attributes
    ----------
    mode : PoleMode
        How poles are specified.
    fixed_poles_hz : list[float]
        Exact pole frequencies (for ``fixed`` mode).
    intervals : list[PoleInterval]
        Pole intervals (for ``intervals`` mode).
    min_separation_hz : float
        Minimum spacing between poles.
    min_distance_from_target_hz : float
        Minimum distance from any pole to any target frequency.
    allowed_band_hz : tuple[float, float] | None
        Overall allowed band for pole placement.
    """

    mode: PoleMode = PoleMode.AUTO
    fixed_poles_hz: list[float] = Field(default_factory=list)
    intervals: list[PoleInterval] = Field(default_factory=list)
    min_separation_hz: float = Field(default=100.0e3, ge=0.0)
    min_distance_from_target_hz: float = Field(default=50.0e3, ge=0.0)
    allowed_band_hz: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> PoleSpec:
        if self.mode == PoleMode.FIXED and not self.fixed_poles_hz:
            raise ValueError("fixed mode requires at least one pole in fixed_poles_hz")
        if self.mode == PoleMode.INTERVALS and not self.intervals:
            raise ValueError("intervals mode requires at least one interval")
        for f in self.fixed_poles_hz:
            if f <= 0.0:
                raise ValueError(f"Pole frequency must be positive, got {f}")
        if self.allowed_band_hz is not None:
            lo, hi = self.allowed_band_hz
            if lo >= hi:
                raise ValueError(f"allowed_band_hz lower ({lo}) must be < upper ({hi})")
        return self


class TopologySearchSpec(BaseModel, frozen=True):
    """Topology enumeration constraints (spec §6.4).

    Attributes
    ----------
    orientations : list[LOrientation]
        Allowed L-network orientations.
    branch1_cells_min : int
        Minimum Foster cells in branch 1.
    branch1_cells_max : int
        Maximum Foster cells in branch 1.
    branch2_cells_min : int
        Minimum Foster cells in branch 2.
    branch2_cells_max : int
        Maximum Foster cells in branch 2.
    endpoint_series_cap_branch1 : bool
        Allow endpoint series capacitor in branch 1.
    endpoint_series_ind_branch1 : bool
        Allow endpoint series inductor in branch 1.
    endpoint_series_cap_branch2 : bool
        Allow endpoint series capacitor in branch 2.
    endpoint_series_ind_branch2 : bool
        Allow endpoint series inductor in branch 2.
    max_total_reactive_components : int
        Hard cap on total reactive elements.
    complexity_penalty : float
        Soft penalty weight per component for optimization.
    pole_spec : PoleSpec
        Pole placement specification.
    """

    orientations: list[LOrientation] = Field(
        default_factory=lambda: [LOrientation.SCHMIDT_SHUNT_THEN_SERIES]
    )
    branch1_cells_min: int = Field(default=1, ge=0)
    branch1_cells_max: int = Field(default=3, ge=0)
    branch2_cells_min: int = Field(default=1, ge=0)
    branch2_cells_max: int = Field(default=3, ge=0)
    endpoint_series_cap_branch1: bool = True
    endpoint_series_ind_branch1: bool = True
    endpoint_series_cap_branch2: bool = True
    endpoint_series_ind_branch2: bool = True
    max_total_reactive_components: int = Field(default=14, ge=1)
    complexity_penalty: float = Field(default=0.02, ge=0.0)
    pole_spec: PoleSpec = Field(default_factory=PoleSpec)

    @model_validator(mode="after")
    def _validate_cells(self) -> TopologySearchSpec:
        if self.branch1_cells_min > self.branch1_cells_max:
            raise ValueError(
                f"branch1_cells_min ({self.branch1_cells_min}) "
                f"must be ≤ branch1_cells_max ({self.branch1_cells_max})"
            )
        if self.branch2_cells_min > self.branch2_cells_max:
            raise ValueError(
                f"branch2_cells_min ({self.branch2_cells_min}) "
                f"must be ≤ branch2_cells_max ({self.branch2_cells_max})"
            )
        if not self.orientations:
            raise ValueError("At least one L-network orientation must be allowed")
        return self
