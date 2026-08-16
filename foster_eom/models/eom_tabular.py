"""Tabular data EOM model."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from foster_eom.domain.eom import ExtrapolationPolicy
from foster_eom.models.base import EOMModel


class TabularEOM(EOMModel):
    """Tabular measured impedance EOM model (spec §7.4)."""

    def __init__(
        self,
        f_hz: np.ndarray,
        z_complex: np.ndarray,
        interpolation: str = "linear",
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        """Initialize the tabular EOM model.

        Parameters
        ----------
        f_hz : np.ndarray
            Frequencies in Hz. Must be 1-D, strictly positive, strictly increasing,
            and have at least 2 points.
        z_complex : np.ndarray
            Complex impedances corresponding to f_hz. Must be 1-D, same length as
            f_hz.
        interpolation : str, default 'linear'
            Interpolation method.  'linear' reduces interpolation overshoot
            compared to 'cubic' but does **not** guarantee global passivity
            or causality of the interpolated impedance.  Rational fitting
            with passivity enforcement (spec §7.5) is the correct
            high-fidelity solution and will be available in a later milestone.
        extrapolation_policy : ExtrapolationPolicy, default ERROR
            Policy for handling frequencies outside the data range.

        Raises
        ------
        ValueError
            If inputs are not 1-D, lengths differ, f_hz is not strictly increasing/positive,
            or data is non-finite.
        """
        super().__init__(extrapolation_policy)

        f_arr = np.asarray(f_hz, dtype=np.float64)
        z_arr = np.asarray(z_complex, dtype=np.complex128)

        if f_arr.ndim != 1 or z_arr.ndim != 1:
            raise ValueError("f_hz and z_complex must be 1-D arrays.")
        if len(f_arr) != len(z_arr):
            raise ValueError("f_hz and z_complex must have the same length.")
        if len(f_arr) < 2:
            raise ValueError("Tabular data requires at least 2 points.")
        if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(z_arr)):
            raise ValueError("Tabular data must contain only finite values.")
        if not np.all(f_arr > 0.0):
            raise ValueError("Frequencies must be strictly positive.")
        if not np.all(np.diff(f_arr) > 0.0):
            raise ValueError("Frequencies must be strictly increasing with no duplicates.")
        if interpolation not in ("linear", "cubic", "nearest"):
            raise ValueError("Interpolation must be 'linear', 'cubic', or 'nearest'.")

        self.f_hz = f_arr
        self.z_complex = z_arr
        self.interpolation = interpolation

        self._validity_hz = (float(f_arr[0]), float(f_arr[-1]))

        # We interpolate real and imaginary parts independently.
        # scipy interp1d accepts complex arrays and handles them component-wise.
        #
        # NOTE: Neither linear nor cubic interpolation of Re(Z)/Im(Z)
        # guarantees passivity (Re(Z) >= 0) or causality of the
        # interpolated function.  Linear reduces spline overshoot but
        # is not a substitute for rational/passivity-enforced modeling
        # (spec §7.5), which is a later milestone.
        self._interp = interp1d(
            self.f_hz,
            self.z_complex,
            kind=self.interpolation,
            bounds_error=False,
            fill_value="extrapolate",
        )

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Evaluate impedance."""
        result = self._interp(f_hz)
        return complex(result) if np.isscalar(f_hz) else result

    def validity_range(self) -> tuple[float, float] | None:
        """Return the range of the provided tabular data."""
        return self._validity_hz

    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata."""
        return {
            "model_type": "tabular",
            "interpolation": self.interpolation,
            "data_points": len(self.f_hz),
            "f_min_hz": self._validity_hz[0],
            "f_max_hz": self._validity_hz[1],
        }
