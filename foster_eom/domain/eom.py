"""EOM model reference for the project spec.

This module defines a lightweight reference structure for EOM models inside the
project schema.  The full EOM model implementations (ideal, lossy, mBVD,
tabular, rational) live in ``foster_eom.models`` and are built in Prompt 02.

For the project YAML, we store the model *definition* — sufficient to
reconstruct the model at load time.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field, model_validator


class EOMModelType(enum.StrEnum):
    """EOM model types (spec §7)."""

    IDEAL_CAPACITOR = "ideal_capacitor"
    LOSSY_CAPACITOR = "lossy_capacitor"
    MBVD = "mbvd"
    TABULAR = "tabular"
    RATIONAL = "rational"
    MEASURED = "measured"


class ExtrapolationPolicy(enum.StrEnum):
    """How to handle frequency queries outside the validity range."""

    ERROR = "error"
    WARN = "warn"
    CLAMP = "clamp"
    ALLOW = "allow"


class MotionalBranch(BaseModel, frozen=True):
    """Single motional RLC branch of an mBVD model.

    Attributes
    ----------
    rm_ohm : float
        Motional resistance (must be ≥ 0).
    lm_h : float
        Motional inductance (must be > 0).
    cm_f : float
        Motional capacitance (must be > 0).
    """

    rm_ohm: float = Field(ge=0.0)
    lm_h: float = Field(gt=0.0)
    cm_f: float = Field(gt=0.0)


class EOMModelSpec(BaseModel, frozen=True):
    """EOM model definition for project persistence.

    The ``model_type`` discriminates which fields are relevant.
    Full electrical models are instantiated by ``foster_eom.models`` from
    these specifications.

    Attributes
    ----------
    model_type : EOMModelType
        Type of EOM model.
    name : str
        Human-readable label (e.g. ``"SYNTHETIC_TEST_ONLY"``).
    validity_hz : tuple[float, float] | None
        Frequency validity range [f_min, f_max] in Hz.

    Ideal capacitor fields
    ----------------------
    c0_f : float | None
        Static capacitance (F).

    Lossy capacitor fields
    ----------------------
    rs_ohm : float | None
        Series electrode/lead resistance (Ω).
    ls_h : float | None
        Series lead inductance (H).
    g0_s : float | None
        Dielectric conductance (S).

    mBVD fields
    -----------
    motional_branches : list[MotionalBranch]
        Motional RLC branches.

    Tabular fields
    --------------
    data_file : str | None
        Path to measurement data file.
    data_format : str | None
        Format descriptor (csv, s1p, etc.)
    """

    model_type: EOMModelType
    name: str = ""
    validity_hz: tuple[float, float] | None = None
    extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR

    # Common capacitor fields
    c0_f: float | None = Field(default=None, gt=0.0)
    g0_s: float | None = Field(default=None, ge=0.0)

    # Series parasitics
    rs_ohm: float | None = Field(default=None, ge=0.0)
    ls_h: float | None = Field(default=None, ge=0.0)

    # mBVD motional branches
    motional_branches: list[MotionalBranch] = Field(default_factory=list)

    # Tabular model reference
    data_file: str | None = None
    data_format: str | None = None

    @model_validator(mode="after")
    def _validate_type_fields(self) -> EOMModelSpec:
        if self.model_type == EOMModelType.IDEAL_CAPACITOR:
            if self.c0_f is None:
                raise ValueError("ideal_capacitor model requires c0_f")
        elif self.model_type == EOMModelType.LOSSY_CAPACITOR:
            if self.c0_f is None:
                raise ValueError("lossy_capacitor model requires c0_f")
        elif self.model_type == EOMModelType.MBVD:
            if self.c0_f is None:
                raise ValueError("mbvd model requires c0_f")
        elif self.model_type == EOMModelType.TABULAR and self.data_file is None:
            raise ValueError("tabular model requires data_file")

        if self.validity_hz is not None:
            lo, hi = self.validity_hz
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError("validity_hz bounds must be positive")
            if lo >= hi:
                raise ValueError(f"validity_hz lower ({lo}) must be < upper ({hi})")
        return self
