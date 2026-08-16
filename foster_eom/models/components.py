"""Primitive component models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from foster_eom.domain.eom import ExtrapolationPolicy
from foster_eom.models.base import OnePortModel


class IdealResistor(OnePortModel):
    """Ideal resistor (Z = R)."""

    def __init__(self, r_ohm: float) -> None:
        super().__init__()
        if r_ohm < 0.0:
            raise ValueError("Resistance must be non-negative.")
        self.r_ohm = r_ohm

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        if np.isscalar(f_hz):
            return complex(self.r_ohm, 0.0)
        return np.full_like(f_hz, complex(self.r_ohm, 0.0), dtype=np.complex128)

    def validity_range(self) -> tuple[float, float] | None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {"model_type": "ideal_resistor", "r_ohm": self.r_ohm}


class IdealInductor(OnePortModel):
    """Ideal inductor (Z = jwL)."""

    def __init__(self, l_h: float) -> None:
        super().__init__()
        if l_h <= 0.0:
            raise ValueError("Inductance must be strictly positive.")
        self.l_h = l_h

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        return 1j * 2.0 * np.pi * f_hz * self.l_h

    def validity_range(self) -> tuple[float, float] | None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {"model_type": "ideal_inductor", "l_h": self.l_h}


class IdealCapacitor(OnePortModel):
    """Ideal capacitor (Z = 1 / jwC)."""

    def __init__(self, c_f: float) -> None:
        super().__init__()
        if c_f <= 0.0:
            raise ValueError("Capacitance must be strictly positive.")
        self.c_f = c_f

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        omega = 2.0 * np.pi * f_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / (1j * omega * self.c_f)

    def _y_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        omega = 2.0 * np.pi * f_hz
        return 1j * omega * self.c_f

    def validity_range(self) -> tuple[float, float] | None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {"model_type": "ideal_capacitor", "c_f": self.c_f}


class LumpedLossyInductor(OnePortModel):
    """Lossy lumped inductor (spec §9.1).

    Topology: (R_dcr + jwL) || 1/(jwC_par)
    """

    def __init__(
        self,
        l_h: float,
        r_dcr_ohm: float = 0.0,
        c_par_f: float = 0.0,
        validity_hz: tuple[float, float] | None = None,
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        super().__init__(extrapolation_policy)
        if l_h <= 0.0:
            raise ValueError("Inductance must be strictly positive.")
        if r_dcr_ohm < 0.0:
            raise ValueError("DCR must be non-negative.")
        if c_par_f < 0.0:
            raise ValueError("Parallel capacitance must be non-negative.")
        if validity_hz is not None:
            if validity_hz[0] <= 0.0 or validity_hz[0] >= validity_hz[1]:
                raise ValueError("Validity bounds must be positive and f_min < f_max.")

        self.l_h = l_h
        self.r_dcr_ohm = r_dcr_ohm
        self.c_par_f = c_par_f
        self._validity_hz = validity_hz

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        omega = 2.0 * np.pi * f_hz
        z_series = self.r_dcr_ohm + 1j * omega * self.l_h
        y_series = 1.0 / z_series
        y_par = 1j * omega * self.c_par_f
        return 1.0 / (y_series + y_par)

    def validity_range(self) -> tuple[float, float] | None:
        return self._validity_hz

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "model_type": "lossy_inductor",
            "l_h": self.l_h,
            "r_dcr_ohm": self.r_dcr_ohm,
            "c_par_f": self.c_par_f,
        }
        if self._validity_hz is not None:
            meta["validity_hz"] = list(self._validity_hz)
        return meta


class LumpedLossyCapacitor(OnePortModel):
    """Lossy lumped capacitor (spec §9.1).

    Topology: R_esr + jwL_esl + 1/(jwC)
    """

    def __init__(
        self,
        c_f: float,
        r_esr_ohm: float = 0.0,
        l_esl_h: float = 0.0,
        validity_hz: tuple[float, float] | None = None,
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        super().__init__(extrapolation_policy)
        if c_f <= 0.0:
            raise ValueError("Capacitance must be strictly positive.")
        if r_esr_ohm < 0.0:
            raise ValueError("ESR must be non-negative.")
        if l_esl_h < 0.0:
            raise ValueError("ESL must be non-negative.")
        if validity_hz is not None:
            if validity_hz[0] <= 0.0 or validity_hz[0] >= validity_hz[1]:
                raise ValueError("Validity bounds must be positive and f_min < f_max.")

        self.c_f = c_f
        self.r_esr_ohm = r_esr_ohm
        self.l_esl_h = l_esl_h
        self._validity_hz = validity_hz

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        omega = 2.0 * np.pi * f_hz
        z_c = 1.0 / (1j * omega * self.c_f)
        return self.r_esr_ohm + 1j * omega * self.l_esl_h + z_c

    def validity_range(self) -> tuple[float, float] | None:
        return self._validity_hz

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "model_type": "lossy_capacitor",
            "c_f": self.c_f,
            "r_esr_ohm": self.r_esr_ohm,
            "l_esl_h": self.l_esl_h,
        }
        if self._validity_hz is not None:
            meta["validity_hz"] = list(self._validity_hz)
        return meta


class TabularImpedanceComponent(OnePortModel):
    """Tabular measured impedance component model (spec §9.1)."""

    def __init__(
        self,
        f_hz: np.ndarray,
        z_complex: np.ndarray,
        interpolation: str = "linear",
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
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

        self._interp = interp1d(
            self.f_hz,
            self.z_complex,
            kind=self.interpolation,
            bounds_error=False,
            fill_value="extrapolate",
        )

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        result = self._interp(f_hz)
        return complex(result) if np.isscalar(f_hz) else result

    def validity_range(self) -> tuple[float, float] | None:
        return self._validity_hz

    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "tabular_component",
            "interpolation": self.interpolation,
            "data_points": len(self.f_hz),
            "f_min_hz": self._validity_hz[0],
            "f_max_hz": self._validity_hz[1],
        }
