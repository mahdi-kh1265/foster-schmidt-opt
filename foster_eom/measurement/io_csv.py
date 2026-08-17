"""Flexible CSV importer for measured S-parameter / impedance data (Prompt 07)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from foster_eom.measurement.dataset import (
    MeasuredDataset,
    SourceQuantity,
    compute_file_sha256,
)

# ---------------------------------------------------------------------------
# Header alias tables (case-insensitive, whitespace-normalized)
# ---------------------------------------------------------------------------

_FREQ_ALIASES = frozenset(
    {
        "freq",
        "frequency",
        "f",
        "f_hz",
        "f_mhz",
        "f_ghz",
        "f_khz",
    }
)
_S11_RE_ALIASES = frozenset(
    {
        "re_s11",
        "s11_re",
        "real_s11",
        "re(s11)",
    }
)
_S11_IM_ALIASES = frozenset(
    {
        "im_s11",
        "s11_im",
        "imag_s11",
        "im(s11)",
    }
)
_S11_MAG_ALIASES = frozenset(
    {
        "mag_s11",
        "s11_mag",
        "|s11|",
    }
)
_S11_ANG_ALIASES = frozenset(
    {
        "ang_s11",
        "s11_ang",
        "phase_s11",
        "deg_s11",
    }
)
_Z_RE_ALIASES = frozenset(
    {
        "re_z",
        "z_re",
        "real_z",
        "re(z)",
    }
)
_Z_IM_ALIASES = frozenset(
    {
        "im_z",
        "z_im",
        "imag_z",
        "im(z)",
    }
)

# Frequency unit multipliers
_FREQ_MULTIPLIERS: dict[str, float] = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


def _normalize_header(h: str) -> str:
    """Lowercase, strip whitespace, collapse internal whitespace to '_'."""
    return re.sub(r"\s+", "_", h.strip().lower())


def _detect_format(headers: list[str]) -> str:
    """Auto-detect CSV format from normalized header aliases.

    Raises ValueError on ambiguous or unrecognized headers.
    """
    has_freq = any(h in _FREQ_ALIASES for h in headers)
    has_s11_ri = any(h in _S11_RE_ALIASES for h in headers) and any(
        h in _S11_IM_ALIASES for h in headers
    )
    has_s11_ma = any(h in _S11_MAG_ALIASES for h in headers) and any(
        h in _S11_ANG_ALIASES for h in headers
    )
    has_z_ri = any(h in _Z_RE_ALIASES for h in headers) and any(h in _Z_IM_ALIASES for h in headers)

    if not has_freq:
        raise ValueError(
            f"No frequency column found. Recognized aliases: {sorted(_FREQ_ALIASES)}. "
            f"Got headers: {headers}"
        )

    matches = []
    if has_s11_ri:
        matches.append("freq_s11_ri")
    if has_s11_ma:
        matches.append("freq_s11_ma")
    if has_z_ri:
        matches.append("freq_z_ri")

    if len(matches) == 0:
        raise ValueError(
            f"Cannot determine CSV format from headers: {headers}. "
            "Use explicit format= or column_map= parameter."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous CSV headers match multiple formats: {matches}. "
            "Use explicit format= or column_map= parameter."
        )

    return matches[0]


def _find_column(headers: list[str], aliases: frozenset[str], label: str) -> int:
    """Find the index of the first header matching any alias."""
    for i, h in enumerate(headers):
        if h in aliases:
            return i
    raise ValueError(f"No column found for {label}. Aliases: {sorted(aliases)}")


def load_csv(
    path: str | Path,
    *,
    format: str = "auto",
    z_ref_ohm: float = 50.0,
    freq_unit: str = "hz",
    column_map: dict[str, str] | None = None,
    instrument: str = "",
    measurement_plane: str = "EOM external RF connector",
    notes: str = "",
) -> MeasuredDataset:
    """Load measured data from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    format : str
        ``"auto"``, ``"freq_s11_ri"``, ``"freq_s11_ma"``, or ``"freq_z_ri"``.
    z_ref_ohm : float
        Reference impedance in Ohms (default 50).
    freq_unit : str
        Frequency unit: ``"hz"``, ``"khz"``, ``"mhz"``, ``"ghz"``.
    column_map : dict | None
        Explicit column mapping overriding auto-detection.
        Keys: ``"freq"``, ``"s11_re"``, ``"s11_im"``, ``"s11_mag"``, ``"s11_ang"``,
        ``"z_re"``, ``"z_im"``.  Values: column header strings.
    instrument, measurement_plane, notes : str
        Provenance fields.

    Returns
    -------
    MeasuredDataset

    Raises
    ------
    ValueError
        On ambiguous headers, malformed data, NaN, duplicates, or wrong column count.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    freq_unit_lower = freq_unit.lower()
    if freq_unit_lower not in _FREQ_MULTIPLIERS:
        raise ValueError(
            f"Unknown freq_unit '{freq_unit}'. Valid: {list(_FREQ_MULTIPLIERS.keys())}"
        )
    freq_mult = _FREQ_MULTIPLIERS[freq_unit_lower]

    # Read lines, skip comments
    lines: list[str] = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("!"):
                lines.append(stripped)

    if len(lines) < 2:
        raise ValueError("CSV must contain a header row and at least one data row.")

    # Parse header
    raw_headers = lines[0].split(",")
    headers = [_normalize_header(h) for h in raw_headers]

    # Apply column_map override if given
    if column_map is not None:
        mapped_headers = []
        inv_map = {_normalize_header(v): _normalize_header(k) for k, v in column_map.items()}
        for h in headers:
            mapped_headers.append(inv_map.get(h, h))
        headers = mapped_headers

    # Determine format
    fmt = format.lower()
    if fmt == "auto":
        fmt = _detect_format(headers)

    # Parse data
    data_rows: list[list[float]] = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        try:
            row = [float(x.strip()) for x in parts]
        except ValueError as exc:
            raise ValueError(f"CSV line {i}: cannot parse as floats: {exc}") from exc
        data_rows.append(row)

    if len(data_rows) < 1:
        raise ValueError("CSV contains no data rows.")

    data = np.array(data_rows, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("CSV data contains NaN or Inf values.")

    # Extract columns by format
    freq_idx = _find_column(headers, _FREQ_ALIASES, "frequency")
    f_hz = data[:, freq_idx] * freq_mult

    if fmt == "freq_s11_ri":
        re_idx = _find_column(headers, _S11_RE_ALIASES, "Re(S11)")
        im_idx = _find_column(headers, _S11_IM_ALIASES, "Im(S11)")
        s11 = data[:, re_idx] + 1j * data[:, im_idx]
        source_q = SourceQuantity.S11
    elif fmt == "freq_s11_ma":
        mag_idx = _find_column(headers, _S11_MAG_ALIASES, "Mag(S11)")
        ang_idx = _find_column(headers, _S11_ANG_ALIASES, "Ang(S11)")
        s11 = data[:, mag_idx] * np.exp(1j * np.deg2rad(data[:, ang_idx]))
        source_q = SourceQuantity.S11
    elif fmt == "freq_z_ri":
        zr_idx = _find_column(headers, _Z_RE_ALIASES, "Re(Z)")
        zi_idx = _find_column(headers, _Z_IM_ALIASES, "Im(Z)")
        z = data[:, zr_idx] + 1j * data[:, zi_idx]
        source_q = SourceQuantity.Z
    else:
        raise ValueError(
            f"Unknown format '{format}'. Valid: 'auto', 'freq_s11_ri', 'freq_s11_ma', 'freq_z_ri'."
        )

    sha = compute_file_sha256(p)

    if source_q == SourceQuantity.Z:
        return MeasuredDataset.from_z(
            f_hz=f_hz,
            z=z,  # type: ignore[possibly-undefined]
            z_ref_ohm=z_ref_ohm,
            source_file=p.name,
            source_sha256=sha,
            source_format="csv",
            instrument=instrument,
            measurement_plane=measurement_plane,
            notes=notes,
        )
    else:
        return MeasuredDataset.from_s11(
            f_hz=f_hz,
            s11=s11,
            z_ref_ohm=z_ref_ohm,
            source_file=p.name,
            source_sha256=sha,
            source_format="csv",
            instrument=instrument,
            measurement_plane=measurement_plane,
            notes=notes,
        )
