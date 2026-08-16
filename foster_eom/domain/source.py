"""Source specification (spec §6.2, §3.2).

Defines the RF source model: Thévenin, available-power, or generator-display
convention.  All representations are normalized to a Thévenin RMS phasor and
source impedance internally.
"""

from __future__ import annotations

import enum
import math

from pydantic import BaseModel, Field, model_validator

from foster_eom.units import (
    available_power_to_vth_rms,
    dbm_to_w,
    deg_to_rad,
    generator_display_to_vth_rms,
)


class SourceMode(enum.StrEnum):
    """Source specification modes (spec §3.2)."""

    THEVENIN = "thevenin"
    AVAILABLE_POWER = "available_power"
    GENERATOR_INTO_Z0 = "generator_into_z0"


class SourceSpec(BaseModel, frozen=True):
    """RF source specification.

    Supports three equivalent input modes.  All are normalized to a Thévenin
    RMS voltage and source impedance via :meth:`vth_rms` and the stored
    ``z_source_*`` fields.

    Attributes
    ----------
    mode : SourceMode
        How the user specifies the source level.
    z_source_real_ohm : float
        Real part of source impedance (must be > 0).
    z_source_imag_ohm : float
        Imaginary part of source impedance (default 0).
    z_ref_ohm : float
        Reference impedance for S11/reflection (default 50 Ω, must be > 0).
    phase_deg : float
        Source phase in degrees (default 0).
    insertion_loss_db : float
        Optional insertion loss between source and matcher input (default 0).

    Mode-specific fields
    --------------------
    available_power_dbm : float | None
        Available power in dBm (for ``available_power`` mode).
    available_power_w : float | None
        Available power in watts (alternative to dBm).
    thevenin_vrms : float | None
        Thévenin RMS voltage (for ``thevenin`` mode).
    thevenin_vpp : float | None
        Thévenin peak-to-peak voltage (alternative).
    generator_display_v : float | None
        Generator display voltage (for ``generator_into_z0`` mode).
    generator_display_convention : str | None
        One of ``rms_into_z0``, ``vpp_into_z0``, ``peak_into_z0``.
    """

    mode: SourceMode

    z_source_real_ohm: float = Field(default=50.0, gt=0.0)
    z_source_imag_ohm: float = Field(default=0.0)
    z_ref_ohm: float = Field(default=50.0, gt=0.0)
    phase_deg: float = Field(default=0.0)
    insertion_loss_db: float = Field(default=0.0, ge=0.0)

    # Available-power mode
    available_power_dbm: float | None = None
    available_power_w: float | None = None

    # Thévenin mode
    thevenin_vrms: float | None = None
    thevenin_vpp: float | None = None

    # Generator display mode
    generator_display_v: float | None = None
    generator_display_convention: str | None = None

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> SourceSpec:
        """Ensure the mode-specific fields are provided."""
        if self.mode == SourceMode.AVAILABLE_POWER:
            if self.available_power_dbm is None and self.available_power_w is None:
                raise ValueError(
                    "available_power mode requires available_power_dbm or available_power_w"
                )
        elif self.mode == SourceMode.THEVENIN:
            if self.thevenin_vrms is None and self.thevenin_vpp is None:
                raise ValueError("thevenin mode requires thevenin_vrms or thevenin_vpp")
        elif self.mode == SourceMode.GENERATOR_INTO_Z0:
            if self.generator_display_v is None:
                raise ValueError("generator_into_z0 mode requires generator_display_v")
            if self.generator_display_convention is None:
                raise ValueError("generator_into_z0 mode requires generator_display_convention")
        return self

    @property
    def z_source(self) -> complex:
        """Complex source impedance in ohms."""
        return complex(self.z_source_real_ohm, self.z_source_imag_ohm)

    @property
    def phase_rad(self) -> float:
        """Source phase in radians."""
        return deg_to_rad(self.phase_deg)

    @property
    def vth_rms(self) -> float:
        """Normalized Thévenin RMS open-circuit voltage in volts.

        This is the single internal representation regardless of input mode.
        """
        if self.mode == SourceMode.AVAILABLE_POWER:
            if self.available_power_w is not None:
                p_w = self.available_power_w
            else:
                assert self.available_power_dbm is not None
                p_w = dbm_to_w(self.available_power_dbm)
            return available_power_to_vth_rms(p_w, self.z_source_real_ohm)

        elif self.mode == SourceMode.THEVENIN:
            if self.thevenin_vrms is not None:
                return self.thevenin_vrms
            else:
                assert self.thevenin_vpp is not None
                # Vpp → Vpeak = Vpp/2 → Vrms = Vpeak/sqrt(2) = Vpp/(2*sqrt(2))
                return self.thevenin_vpp / (2.0 * math.sqrt(2.0))

        else:  # GENERATOR_INTO_Z0
            assert self.generator_display_v is not None
            assert self.generator_display_convention is not None
            return generator_display_to_vth_rms(
                self.generator_display_v,
                self.generator_display_convention,
                self.z_source_real_ohm,
            )

    @property
    def vth_phasor(self) -> complex:
        """Thévenin RMS phasor (magnitude + phase)."""
        import cmath

        return cmath.rect(self.vth_rms, self.phase_rad)
