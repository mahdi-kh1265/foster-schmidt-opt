"""Modified Butterworth-Van Dyke (mBVD) EOM model."""

from __future__ import annotations

from typing import Any

import numpy as np

from foster_eom.domain.eom import ExtrapolationPolicy, MotionalBranch
from foster_eom.models.base import EOMModel


class MBVDModel(EOMModel):
    """Modified Butterworth-Van Dyke EOM model (spec §7.3).

    Y_core = G0 + j * omega * C0 + sum_k(1 / (Rm_k + j * omega * Lm_k + 1 / (j * omega * Cm_k)))
    Z_EOM = Rs + j * omega * Ls + 1 / Y_core
    """

    def __init__(
        self,
        c0_f: float,
        g0_s: float = 0.0,
        rs_ohm: float = 0.0,
        ls_h: float = 0.0,
        motional_branches: list[MotionalBranch] | None = None,
        validity_hz: tuple[float, float] | None = None,
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        """Initialize the mBVD EOM model.

        Parameters
        ----------
        c0_f : float
            Static capacitance in Farads. Must be > 0.
        g0_s : float, default 0.0
            Dielectric conductance in Siemens. Must be >= 0.
        rs_ohm : float, default 0.0
            Series resistance in Ohms. Must be >= 0.
        ls_h : float, default 0.0
            Series inductance in Henrys. Must be >= 0.
        motional_branches : list[MotionalBranch] | None, default None
            List of motional branches.
        validity_hz : tuple[float, float] | None, default None
            Frequency validity range [f_min, f_max] in Hz.
        extrapolation_policy : ExtrapolationPolicy, default ERROR
            Policy for handling frequencies outside the valid range.

        Raises
        ------
        ValueError
            If components are negative or bounds are invalid.
        """
        super().__init__(extrapolation_policy)

        if c0_f <= 0.0:
            raise ValueError("Capacitance c0_f must be strictly positive.")
        if rs_ohm < 0.0 or ls_h < 0.0 or g0_s < 0.0:
            raise ValueError("Parasitic components (Rs, Ls, G0) must be non-negative.")
        if validity_hz is not None and (validity_hz[0] <= 0.0 or validity_hz[0] >= validity_hz[1]):
            raise ValueError("Validity bounds must be positive and f_min < f_max.")

        self.c0_f = c0_f
        self.g0_s = g0_s
        self.rs_ohm = rs_ohm
        self.ls_h = ls_h
        self.motional_branches = motional_branches or []
        self._validity_hz = validity_hz

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Evaluate impedance."""
        omega = 2.0 * np.pi * f_hz

        y_core = self.g0_s + 1j * omega * self.c0_f

        for branch in self.motional_branches:
            with np.errstate(divide="ignore", invalid="ignore"):
                z_m = branch.rm_ohm + 1j * omega * branch.lm_h + 1.0 / (1j * omega * branch.cm_f)
                # Use np.where / np.divide to handle z_m == 0 for both scalar
                # and array paths without raising ZeroDivisionError.
                z_m_arr = np.asarray(z_m, dtype=np.complex128)
                y_m = np.where(z_m_arr == 0j, np.complex128(np.inf), 1.0 / z_m_arr)
                y_core = y_core + y_m

        z_series = self.rs_ohm + 1j * omega * self.ls_h

        with np.errstate(divide="ignore", invalid="ignore"):
            return z_series + (1.0 / y_core)

    def validity_range(self) -> tuple[float, float] | None:
        """Return the specified validity range."""
        return self._validity_hz

    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata."""
        meta: dict[str, Any] = {
            "model_type": "mbvd",
            "c0_f": self.c0_f,
            "g0_s": self.g0_s,
            "rs_ohm": self.rs_ohm,
            "ls_h": self.ls_h,
            "motional_branches": [
                {"rm_ohm": b.rm_ohm, "lm_h": b.lm_h, "cm_f": b.cm_f} for b in self.motional_branches
            ],
        }
        if self._validity_hz is not None:
            meta["validity_hz"] = list(self._validity_hz)
        return meta
