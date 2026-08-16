"""Ideal capacitor EOM model."""

from __future__ import annotations

from typing import Any

import numpy as np

from foster_eom.domain.eom import ExtrapolationPolicy
from foster_eom.models.base import EOMModel


class IdealCapacitorEOM(EOMModel):
    """Ideal capacitor EOM model (spec §7.1).

    Z = 1 / (j * omega * C0)
    """

    def __init__(
        self,
        c0_f: float,
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        """Initialize the ideal capacitor EOM model.

        Parameters
        ----------
        c0_f : float
            Capacitance in Farads. Must be > 0.
        extrapolation_policy : ExtrapolationPolicy
            Policy for handling frequencies outside the valid range.
            (Unused by this model since it has infinite validity, but kept
            for API consistency).

        Raises
        ------
        ValueError
            If c0_f <= 0.
        """
        super().__init__(extrapolation_policy)
        if c0_f <= 0.0:
            raise ValueError("Capacitance c0_f must be strictly positive.")
        self.c0_f = c0_f

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Evaluate impedance."""
        omega = 2.0 * np.pi * f_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / (1j * omega * self.c0_f)

    def _y_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Evaluate admittance."""
        omega = 2.0 * np.pi * f_hz
        return 1j * omega * self.c0_f

    def validity_range(self) -> tuple[float, float] | None:
        """Ideal capacitor has infinite validity (mathematical model)."""
        return None

    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata."""
        return {
            "model_type": "ideal_capacitor",
            "description": "Ideal capacitor EOM model (Z = 1 / j w C0)",
            "c0_f": self.c0_f,
            "mathematical_only": True,
        }
