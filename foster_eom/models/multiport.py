"""Multi-port and Touchstone models (Placeholder for Prompt 08)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from foster_eom.domain.eom import ExtrapolationPolicy


class MultiPortModel(ABC):
    """Base interface for multi-port electrical components."""

    def __init__(
        self, extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR
    ) -> None:
        self.extrapolation_policy = extrapolation_policy

    @abstractmethod
    def z_matrix(self, f_hz: float | np.ndarray) -> np.ndarray:
        """Compute the complex impedance matrix."""
        raise NotImplementedError

    @abstractmethod
    def s_matrix(self, f_hz: float | np.ndarray, z_ref: float = 50.0) -> np.ndarray:
        """Compute the complex scattering matrix."""
        raise NotImplementedError

    def validity_range(self) -> tuple[float, float] | None:
        """Frequency validity range [f_min, f_max] in Hz, or None if infinite."""
        return None

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata about the model."""
        pass


class TouchstoneComponentModel(MultiPortModel):
    """Touchstone/S-parameter component model (Placeholder)."""

    def __init__(
        self,
        file_path: str,
        extrapolation_policy: ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> None:
        super().__init__(extrapolation_policy)
        self.file_path = file_path

    def z_matrix(self, f_hz: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError("Touchstone import is deferred to Prompt 08.")

    def s_matrix(self, f_hz: float | np.ndarray, z_ref: float = 50.0) -> np.ndarray:
        raise NotImplementedError("Touchstone import is deferred to Prompt 08.")

    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "touchstone",
            "file_path": self.file_path,
        }
