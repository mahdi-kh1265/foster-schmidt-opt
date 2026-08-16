"""Component policy specification (spec §6.6).

Defines continuous component limits, allowed manufacturers/series, derating,
dielectric, and catalog realization settings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ContinuousLimits(BaseModel, frozen=True):
    """Allowed continuous component value ranges.

    Attributes
    ----------
    l_min_h : float
        Minimum inductance in henries.
    l_max_h : float
        Maximum inductance in henries.
    c_min_f : float
        Minimum capacitance in farads.
    c_max_f : float
        Maximum capacitance in farads.
    """

    l_min_h: float = Field(default=10.0e-9, gt=0.0)
    l_max_h: float = Field(default=100.0e-6, gt=0.0)
    c_min_f: float = Field(default=0.2e-12, gt=0.0)
    c_max_f: float = Field(default=20.0e-9, gt=0.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> ContinuousLimits:
        if self.l_min_h > self.l_max_h:
            raise ValueError(f"l_min_h ({self.l_min_h}) must be ≤ l_max_h ({self.l_max_h})")
        if self.c_min_f > self.c_max_f:
            raise ValueError(f"c_min_f ({self.c_min_f}) must be ≤ c_max_f ({self.c_max_f})")
        return self


class ComponentPolicy(BaseModel, frozen=True):
    """Component selection and realization policy (spec §6.6).

    Attributes
    ----------
    continuous_limits : ContinuousLimits
        Allowed continuous L/C value ranges.
    capacitor_dielectrics : list[str]
        Allowed capacitor dielectric types (e.g. ``["C0G", "NP0"]``).
    allowed_inductor_families : list[str]
        Allowed inductor series/families.
    allowed_capacitor_families : list[str]
        Allowed capacitor series/families.
    allowed_packages : list[str]
        Allowed component packages.
    allowed_manufacturers : list[str]
        Allowed manufacturer names (empty = all).
    min_inductor_srf_ratio : float
        Minimum ratio of SRF to operating frequency.
    voltage_derating_fraction : float
        Fraction of rated voltage allowed (0-1).
    current_derating_fraction : float
        Fraction of rated current allowed (0-1).
    max_dcr_ohm : float | None
        Maximum allowed DC resistance.
    min_q_at_target : float | None
        Minimum component Q at target frequencies.
    catalog_realization_enabled : bool
        Whether to perform catalog realization.
    allow_series_parallel : bool
        Allow series/parallel part combinations.
    max_parts_per_element : int
        Maximum parts per logical element.
    """

    continuous_limits: ContinuousLimits = Field(default_factory=ContinuousLimits)
    capacitor_dielectrics: list[str] = Field(default_factory=lambda: ["C0G", "NP0"])
    allowed_inductor_families: list[str] = Field(default_factory=list)
    allowed_capacitor_families: list[str] = Field(default_factory=list)
    allowed_packages: list[str] = Field(default_factory=list)
    allowed_manufacturers: list[str] = Field(default_factory=list)
    min_inductor_srf_ratio: float = Field(default=2.0, gt=0.0)
    voltage_derating_fraction: float = Field(default=0.60, gt=0.0, le=1.0)
    current_derating_fraction: float = Field(default=0.60, gt=0.0, le=1.0)
    max_dcr_ohm: float | None = Field(default=None, gt=0.0)
    min_q_at_target: float | None = Field(default=None, gt=0.0)
    catalog_realization_enabled: bool = False
    allow_series_parallel: bool = False
    max_parts_per_element: int = Field(default=1, ge=1)
