"""Base abstractions for electrical models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from foster_eom.domain.eom import ExtrapolationPolicy
from foster_eom.errors import WarningRecord, WarningSeverity


class OnePortModel(ABC):
    """Base interface for a generic one-port electrical component.

    Provides a centralized Template Method pattern for validating frequency inputs,
    enforcing extrapolation policies, and avoiding warning spam, while delegating
    actual mathematical evaluation to subclasses via `_z_impl()` or `_y_impl()`.
    """

    def __init__(
        self, extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR
    ) -> None:
        self.extrapolation_policy = extrapolation_policy
        # Diagnostic-only flag: controls whether the WARN policy has already
        # emitted its one-shot warning for this model instance.  This flag
        # MUST NOT affect numerical output, hashing, serialization, or
        # optimization reproducibility.  It is safe for single-threaded use;
        # for parallel workers each process gets its own model copy.
        self._warned_extrapolation = False

    def z(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Compute the complex impedance at given frequencies.

        Parameters
        ----------
        f_hz : float | np.ndarray
            Frequency or array of frequencies in Hz. Must be strictly positive
            and finite.

        Returns
        -------
        complex | np.ndarray
            Impedance array of same shape as input (or scalar).
        """
        f = self._enforce_validity(f_hz)
        return self._z_impl(f)

    def y(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Compute the complex admittance at given frequencies.

        Parameters
        ----------
        f_hz : float | np.ndarray
            Frequency or array of frequencies in Hz. Must be strictly positive
            and finite.

        Returns
        -------
        complex | np.ndarray
            Admittance array of same shape as input (or scalar).
        """
        f = self._enforce_validity(f_hz)
        return self._y_impl(f)

    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Protected implementation of impedance.

        Subclasses must implement either this or ``_y_impl``.  If neither is
        overridden, a ``NotImplementedError`` is raised at runtime to prevent
        infinite mutual recursion between the two defaults.
        """
        if type(self)._y_impl is OnePortModel._y_impl:
            raise NotImplementedError(f"{type(self).__name__} must override _z_impl or _y_impl.")
        y_val = self._y_impl(f_hz)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / y_val

    def _y_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        """Protected implementation of admittance.

        Subclasses must implement either this or ``_z_impl``.  If neither is
        overridden, a ``NotImplementedError`` is raised at runtime to prevent
        infinite mutual recursion between the two defaults.
        """
        if type(self)._z_impl is OnePortModel._z_impl:
            raise NotImplementedError(f"{type(self).__name__} must override _z_impl or _y_impl.")
        z_val = self._z_impl(f_hz)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / z_val

    def _enforce_validity(self, f_hz: float | np.ndarray) -> float | np.ndarray:
        """Validate input frequencies against positivity and model validity range.

        Applies the ExtrapolationPolicy if frequencies are out of bounds.

        Raises
        ------
        ValueError
            If inputs are not positive or finite.
        ModelValidityError
            If inputs are out of bounds and policy is ERROR.
        """
        f_arr = np.asarray(f_hz, dtype=np.float64)

        if not np.all(np.isfinite(f_arr)):
            raise ValueError("Frequencies must be finite.")
        if not np.all(f_arr > 0.0):
            raise ValueError("Frequencies must be strictly positive.")

        validity = self.validity_range()
        if validity is None:
            return f_hz  # Mathematical model with infinite validity

        f_min, f_max = validity
        mask_out = (f_arr < f_min) | (f_arr > f_max)

        if not np.any(mask_out):
            return f_hz

        # Extrapolation occurred
        if self.extrapolation_policy == ExtrapolationPolicy.ERROR:
            from foster_eom.errors import ModelValidityError

            raise ModelValidityError(
                f"Frequencies outside model validity range [{f_min}, {f_max}] Hz."
            )

        if self.extrapolation_policy == ExtrapolationPolicy.WARN:
            if not self._warned_extrapolation:
                record = WarningRecord(
                    code="MODEL_EXTRAPOLATION",
                    severity=WarningSeverity.WARNING,
                    message=f"Model evaluated outside valid range [{f_min}, {f_max}] Hz.",
                    recommended_action="Check sweep bounds or change extrapolation policy.",
                )
                # Emit standard python warning, but we use WarningRecord string representation.
                warnings.warn(str(record), UserWarning, stacklevel=3)
                self._warned_extrapolation = True
            return f_hz

        if self.extrapolation_policy == ExtrapolationPolicy.CLAMP:
            clamped = np.clip(f_arr, f_min, f_max)
            return float(clamped) if np.isscalar(f_hz) else clamped

        # ALLOW
        return f_hz

    def validity_range(self) -> tuple[float, float] | None:
        """Frequency validity range [f_min, f_max] in Hz, or None if infinite."""
        return None

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata about the model.

        Not intended as the sole serialization mechanism.
        """
        pass

    def parameter_covariance(self) -> np.ndarray | None:
        """Return the parameter covariance matrix, if the model was fitted.

        Returns
        -------
        np.ndarray | None
            Covariance matrix, or None if unknown/inapplicable.
        """
        return None

    def reset_warnings(self) -> None:
        """Reset the warning suppression flag.

        Call this to re-enable the one-shot WARN extrapolation warning,
        for example at the start of a new optimization run or when the
        model is reused in a different context.
        """
        self._warned_extrapolation = False


class EOMModel(OnePortModel):
    """Base interface specifically for EOM load models.

    Extends a standard one-port model with optional electro-optic transfer
    properties.
    """

    def beta_per_v(self, f_hz: float | np.ndarray) -> complex | np.ndarray | None:
        """Evaluate the optical phase modulation transfer function.

        Returns
        -------
        complex | np.ndarray | None
            Optical phase shift per Volt, or None if not modeled.
        """
        return None
