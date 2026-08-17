"""Measured one-port model for MNA evaluation (Prompt 07).

Wraps ``MeasuredDataset`` as an ``EOMModel`` for use in the optimizer/MNA
pipeline.  The raw measured data is the truth; fitting produces a separate
``FitResult`` and never mutates this model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from foster_eom.domain.eom import ExtrapolationPolicy
from foster_eom.measurement.dataset import MeasuredDataset
from foster_eom.models.base import EOMModel


class MeasuredOnePortModel(EOMModel):
    """Tabulated measured impedance model with interpolation.

    Parameters
    ----------
    dataset : MeasuredDataset
        Immutable measured dataset (never modified).
    interpolation : str
        Interpolation method: ``"linear"`` (default), ``"cubic"``, ``"nearest"``.
    extrapolation_policy : ExtrapolationPolicy
        Policy for out-of-range frequency queries.
    """

    def __init__(
        self,
        dataset: MeasuredDataset,
        interpolation: str = "linear",
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        super().__init__(extrapolation_policy)
        if interpolation not in ("linear", "cubic", "nearest"):
            raise ValueError(
                f"Interpolation must be 'linear', 'cubic', or 'nearest', got '{interpolation}'."
            )

        self._dataset = dataset
        self._interpolation = interpolation
        self._extrapolation_occurred = False

        # Interpolate real and imaginary parts independently
        self._interp = interp1d(
            dataset.f_hz,
            dataset.z_complex,
            kind=interpolation,
            bounds_error=False,
            fill_value="extrapolate",
        )

    @property
    def dataset(self) -> MeasuredDataset:
        """Read-only access to the original measured dataset."""
        return self._dataset

    @property
    def extrapolation_occurred(self) -> bool:
        """Whether any evaluation has queried outside the validity range."""
        return self._extrapolation_occurred

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Evaluate impedance via interpolation."""
        result = self._interp(f_hz)
        return complex(result) if np.isscalar(f_hz) else result

    def _enforce_validity(self, f_hz: float | np.ndarray) -> float | np.ndarray:
        """Extend base method to always record extrapolation, even with ALLOW."""
        f_arr = np.asarray(f_hz, dtype=np.float64)
        validity = self.validity_range()
        if validity is not None:
            f_min, f_max = validity
            mask_out = (f_arr < f_min) | (f_arr > f_max)
            if np.any(mask_out):
                self._extrapolation_occurred = True
        return super()._enforce_validity(f_hz)

    def validity_range(self) -> tuple[float, float]:
        """Return the measured data frequency span.  Never None."""
        return self._dataset.validity_hz

    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata."""
        return {
            "model_type": "measured",
            "interpolation": self._interpolation,
            "data_points": len(self._dataset.f_hz),
            "f_min_hz": self._dataset.validity_hz[0],
            "f_max_hz": self._dataset.validity_hz[1],
            "z_ref_ohm": self._dataset.z_ref_ohm,
            "source_file": self._dataset.source_file,
            "source_sha256": self._dataset.source_sha256,
            "source_format": self._dataset.source_format,
            "source_quantity": str(self._dataset.source_quantity),
            "instrument": self._dataset.instrument,
            "measurement_plane": self._dataset.measurement_plane,
            "passivity_flags": list(self._dataset.passivity_flags),
        }
