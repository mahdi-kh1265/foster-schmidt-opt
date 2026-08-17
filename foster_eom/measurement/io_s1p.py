"""Touchstone S1P file loader (Prompt 07).

Delegates Touchstone parsing to scikit-rf.  We own validation, provenance,
and conversion into ``MeasuredDataset``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from foster_eom.measurement.dataset import (
    MeasuredDataset,
    compute_file_sha256,
)


def load_s1p(
    path: str | Path,
    *,
    instrument: str = "",
    measurement_plane: str = "EOM external RF connector",
    notes: str = "",
) -> MeasuredDataset:
    """Load a Touchstone ``.s1p`` file into a ``MeasuredDataset``.

    Parameters
    ----------
    path : str | Path
        Path to the ``.s1p`` file.
    instrument : str
        Instrument identifier for provenance.
    measurement_plane : str
        Reference plane for the measurement.
    notes : str
        Optional free-text metadata.

    Returns
    -------
    MeasuredDataset
        Immutable measured dataset with S11 as canonical truth.

    Raises
    ------
    ValueError
        If the file is not a valid one-port Touchstone, or if the reference
        impedance is non-uniform, complex, or non-positive.
    FileNotFoundError
        If the file does not exist.
    """
    import skrf  # type: ignore[import-untyped]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"S1P file not found: {p}")

    # scikit-rf handles all Touchstone parsing (RI/MA/DB, Hz/kHz/MHz/GHz)
    try:
        ntwk = skrf.Network(str(p))
    except Exception as exc:
        raise ValueError(f"scikit-rf could not parse '{p.name}': {exc}") from exc

    # Validate one-port
    if ntwk.nports != 1:
        raise ValueError(f"Expected 1-port Touchstone, got {ntwk.nports}-port in '{p.name}'.")

    # Extract data
    f_hz = np.asarray(ntwk.f, dtype=np.float64)
    s11 = ntwk.s[:, 0, 0].astype(np.complex128)

    # Reference impedance validation
    z0 = ntwk.z0  # shape (n_freq, n_ports)
    if z0.shape[1] != 1:
        raise ValueError(f"Expected 1-port z0, got shape {z0.shape}.")

    z0_flat = z0[:, 0]

    # Check for complex impedance
    if np.any(np.abs(z0_flat.imag) > 1e-10):
        raise ValueError(
            "Complex reference impedance not supported. "
            "Renormalize to a real impedance before import."
        )

    # Check for non-uniform impedance
    z0_real = z0_flat.real
    if np.ptp(z0_real) > 1e-6:
        raise ValueError(
            "Non-uniform reference impedance not supported; "
            "renormalize to a constant impedance (e.g. 50 Ω) before import."
        )

    z_ref = float(z0_real[0])
    if z_ref <= 0.0:
        raise ValueError(f"Reference impedance must be > 0, got {z_ref}.")

    # Our validation: finite, increasing, ≥2 points
    if len(f_hz) < 2:
        raise ValueError(f"S1P file must contain at least 2 data points, got {len(f_hz)}.")
    if not np.all(np.isfinite(f_hz)):
        raise ValueError("S1P file contains non-finite frequency values.")
    if not np.all(np.isfinite(s11)):
        raise ValueError("S1P file contains non-finite S11 values.")
    if not np.all(f_hz > 0.0):
        raise ValueError("S1P file contains non-positive frequencies.")
    if not np.all(np.diff(f_hz) > 0.0):
        raise ValueError(
            "S1P frequencies are not strictly increasing (possible duplicates or disorder)."
        )

    sha = compute_file_sha256(p)

    return MeasuredDataset.from_s11(
        f_hz=f_hz,
        s11=s11,
        z_ref_ohm=z_ref,
        source_file=p.name,
        source_sha256=sha,
        source_format="s1p",
        instrument=instrument,
        measurement_plane=measurement_plane,
        notes=notes,
    )
