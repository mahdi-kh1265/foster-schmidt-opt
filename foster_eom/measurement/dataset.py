"""Measured one-port dataset (Prompt 07).

Immutable container for measured S11 / impedance data with full provenance,
numerically safe S11↔Z↔Y conversion, and passivity diagnostics.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceQuantity(enum.StrEnum):
    """Which physical quantity was directly measured."""

    S11 = "S11"
    Z = "Z"


# ---------------------------------------------------------------------------
# Safe S11 ↔ Z ↔ Y conversion
# ---------------------------------------------------------------------------

# Points where |1 - S11| < _SINGULAR_TOL are flagged as singular for Z conversion.
_SINGULAR_TOL = 1e-12


def _s11_to_z(s11: np.ndarray, z_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert S11 → Z with singularity detection.

    Returns
    -------
    z : np.ndarray
        Complex impedance.  Singular points (S11 ≈ +1) are set to ``np.inf+0j``.
    singular_mask : np.ndarray
        Boolean mask; True where the conversion is numerically unreliable.
    """
    denom = 1.0 - s11
    singular = np.abs(denom) < _SINGULAR_TOL
    safe_denom = np.where(singular, 1.0, denom)  # avoid division by zero
    z = z_ref * (1.0 + s11) / safe_denom
    z = np.where(singular, np.inf + 0j, z)
    return z, singular


def _z_to_s11(z: np.ndarray, z_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert Z → S11 with singularity detection.

    Returns
    -------
    s11 : np.ndarray
        Complex S11.  Singular points (Z + z_ref ≈ 0) are set to ``-1+0j``.
    singular_mask : np.ndarray
        Boolean mask; True where Z + z_ref ≈ 0 (non-physical for passive device).
    """
    numer = z - z_ref
    denom = z + z_ref
    singular = np.abs(denom) < _SINGULAR_TOL
    safe_denom = np.where(singular, 1.0, denom)
    s11 = numer / safe_denom
    s11 = np.where(singular, -1.0 + 0j, s11)
    return s11, singular


def _z_to_y(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Z → Y with singularity detection.

    Returns
    -------
    y : np.ndarray
        Complex admittance.  Singular points (Z ≈ 0) are set to ``np.inf+0j``.
    singular_mask : np.ndarray
        Boolean mask; True where |Z| < tolerance.
    """
    singular = np.abs(z) < _SINGULAR_TOL
    safe_z = np.where(singular, 1.0, z)
    y = 1.0 / safe_z
    y = np.where(singular, np.inf + 0j, y)
    return y, singular


# ---------------------------------------------------------------------------
# Passivity diagnostics
# ---------------------------------------------------------------------------

_PASSIVITY_S11_TOL = 1e-3
_PASSIVITY_REZ_TOL = -0.1  # Ω


def _check_passivity(
    s11: np.ndarray, z: np.ndarray, f_hz: np.ndarray, z_singular: np.ndarray
) -> tuple[str, ...]:
    """Flag passivity violations; warn, never reject."""
    flags: list[str] = []
    s11_mag = np.abs(s11)
    violating = s11_mag > 1.0 + _PASSIVITY_S11_TOL
    if np.any(violating):
        worst_idx = int(np.argmax(s11_mag))
        flags.append(
            f"|S11|={s11_mag[worst_idx]:.6f} > 1+tol at f={f_hz[worst_idx]:.0f} Hz "
            f"({int(np.sum(violating))} points)"
        )
    # Check Re(Z) < threshold only at non-singular points
    valid_z = ~z_singular & np.isfinite(z.real)
    if np.any(valid_z):
        neg_rez = z[valid_z].real < _PASSIVITY_REZ_TOL
        if np.any(neg_rez):
            worst_real = float(np.min(z[valid_z].real))
            flags.append(
                f"Re(Z)={worst_real:.3f} Ω < {_PASSIVITY_REZ_TOL} Ω ({int(np.sum(neg_rez))} points)"
            )
    return tuple(flags)


# ---------------------------------------------------------------------------
# MeasuredDataset
# ---------------------------------------------------------------------------


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Defensive copy + set read-only."""
    out = np.array(arr, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class MeasuredDataset:
    """Immutable measured one-port dataset.

    Attributes
    ----------
    f_hz : np.ndarray
        Frequencies in Hz.  1-D, strictly increasing, all > 0.  Read-only.
    s11_complex : np.ndarray
        Complex S11.  This is the canonical stored representation.  Read-only.
    z_complex : np.ndarray
        Derived Z = z_ref * (1+S11)/(1-S11).  Singular points = inf.  Read-only.
    y_complex : np.ndarray
        Derived Y = 1/Z.  Singular points = inf.  Read-only.
    z_ref_ohm : float
        Scalar real reference impedance > 0.
    source_quantity : SourceQuantity
        Which array is the raw measurement truth.
    validity_hz : tuple[float, float]
        (f_hz[0], f_hz[-1]).
    z_singular_mask : np.ndarray
        Boolean mask; True where S11→Z conversion is singular.
    y_singular_mask : np.ndarray
        Boolean mask; True where Z→Y conversion is singular.
    source_file : str | None
        Original filename for provenance.
    source_sha256 : str | None
        Hex SHA-256 of source file bytes.
    source_format : str
        ``"s1p"`` or ``"csv"``.
    instrument : str
        Instrument identifier (e.g. ``"LibreVNA"``).
    measurement_plane : str
        Reference plane for the measurement (e.g. ``"EOM external RF connector"``).
    notes : str
        Free-text metadata.
    passivity_flags : tuple[str, ...]
        Non-empty if passivity violations detected.
    """

    f_hz: np.ndarray
    s11_complex: np.ndarray
    z_complex: np.ndarray
    y_complex: np.ndarray
    z_ref_ohm: float
    source_quantity: SourceQuantity
    validity_hz: tuple[float, float]
    z_singular_mask: np.ndarray
    y_singular_mask: np.ndarray
    source_file: str | None
    source_sha256: str | None
    source_format: str
    instrument: str
    measurement_plane: str
    notes: str
    passivity_flags: tuple[str, ...]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_s11(
        f_hz: np.ndarray,
        s11: np.ndarray,
        z_ref_ohm: float = 50.0,
        *,
        source_file: str | None = None,
        source_sha256: str | None = None,
        source_format: str = "s1p",
        instrument: str = "",
        measurement_plane: str = "EOM external RF connector",
        notes: str = "",
    ) -> MeasuredDataset:
        """Construct from raw S11 data."""
        f_arr, s11_arr = _validate_arrays(f_hz, s11, z_ref_ohm)
        z, z_sing = _s11_to_z(s11_arr, z_ref_ohm)
        y, y_sing = _z_to_y(z)
        flags = _check_passivity(s11_arr, z, f_arr, z_sing)
        return MeasuredDataset(
            f_hz=_freeze(f_arr),
            s11_complex=_freeze(s11_arr),
            z_complex=_freeze(z),
            y_complex=_freeze(y),
            z_ref_ohm=z_ref_ohm,
            source_quantity=SourceQuantity.S11,
            validity_hz=(float(f_arr[0]), float(f_arr[-1])),
            z_singular_mask=_freeze(z_sing),
            y_singular_mask=_freeze(y_sing),
            source_file=source_file,
            source_sha256=source_sha256,
            source_format=source_format,
            instrument=instrument,
            measurement_plane=measurement_plane,
            notes=notes,
            passivity_flags=flags,
        )

    @staticmethod
    def from_z(
        f_hz: np.ndarray,
        z: np.ndarray,
        z_ref_ohm: float = 50.0,
        *,
        source_file: str | None = None,
        source_sha256: str | None = None,
        source_format: str = "csv",
        instrument: str = "",
        measurement_plane: str = "EOM external RF connector",
        notes: str = "",
    ) -> MeasuredDataset:
        """Construct from raw impedance data (Z is the truth)."""
        f_arr = np.asarray(f_hz, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.complex128)
        _validate_freq(f_arr)
        if z_arr.ndim != 1 or len(z_arr) != len(f_arr):
            raise ValueError("z must be 1-D and same length as f_hz.")

        # Convert Z → S11 for canonical storage
        s11, _s11_sing = _z_to_s11(z_arr, z_ref_ohm)
        # Recompute Z from S11 for consistency (may differ at singular points)
        _z_from_s11, z_sing = _s11_to_z(s11, z_ref_ohm)
        y, y_sing = _z_to_y(z_arr)
        flags = _check_passivity(s11, z_arr, f_arr, z_sing)

        return MeasuredDataset(
            f_hz=_freeze(f_arr),
            s11_complex=_freeze(s11),
            z_complex=_freeze(z_arr),  # use original Z as truth
            y_complex=_freeze(y),
            z_ref_ohm=z_ref_ohm,
            source_quantity=SourceQuantity.Z,
            validity_hz=(float(f_arr[0]), float(f_arr[-1])),
            z_singular_mask=_freeze(z_sing),
            y_singular_mask=_freeze(y_sing),
            source_file=source_file,
            source_sha256=source_sha256,
            source_format=source_format,
            instrument=instrument,
            measurement_plane=measurement_plane,
            notes=notes,
            passivity_flags=flags,
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_freq(f_arr: np.ndarray) -> None:
    """Validate frequency array invariants."""
    if f_arr.ndim != 1:
        raise ValueError("f_hz must be 1-D.")
    if len(f_arr) < 2:
        raise ValueError("At least 2 frequency points required.")
    if not np.all(np.isfinite(f_arr)):
        raise ValueError("f_hz contains NaN or Inf values.")
    if not np.all(f_arr > 0.0):
        raise ValueError("All frequencies must be strictly positive.")
    if not np.all(np.diff(f_arr) > 0.0):
        raise ValueError("Frequencies must be strictly increasing (no duplicates).")


def _validate_arrays(
    f_hz: np.ndarray, s11: np.ndarray, z_ref_ohm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize input arrays for MeasuredDataset construction."""
    f_arr = np.asarray(f_hz, dtype=np.float64)
    s11_arr = np.asarray(s11, dtype=np.complex128)

    _validate_freq(f_arr)

    if s11_arr.ndim != 1 or len(s11_arr) != len(f_arr):
        raise ValueError("s11 must be 1-D and same length as f_hz.")
    if not np.all(np.isfinite(s11_arr)):
        raise ValueError("s11 contains NaN or Inf values.")

    if not isinstance(z_ref_ohm, (int, float)):
        raise ValueError(f"z_ref_ohm must be a scalar real number, got {type(z_ref_ohm).__name__}.")
    if isinstance(z_ref_ohm, complex):
        raise ValueError("Complex reference impedance not supported.")
    if z_ref_ohm <= 0.0:
        raise ValueError(f"z_ref_ohm must be > 0, got {z_ref_ohm}.")

    return f_arr, s11_arr


def compute_file_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
