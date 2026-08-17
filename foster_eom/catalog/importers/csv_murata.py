"""Murata CSV adapter (Prompt 08).

Genuine parsing for Murata GRM/GCM series capacitor catalogs. Explicit CSV
columns are primary data; part-number decoding is fallback/cross-check only.
"""

from __future__ import annotations

import re
from typing import Any

from foster_eom.catalog.importers.csv_generic import GenericCSVImporter

# Murata part-number decoder (e.g. GRM155R71C104KA88)
# Only used as fallback when CSV columns are missing
_MURATA_PN_RE = re.compile(
    r"^(GRM|GCM|GRJ|GRT)(\d{3})"  # series + size code
    r"([A-Z])"  # height code
    r"(\d)"  # temp coeff code
    r"(\d[A-Z])"  # voltage code
    r"(\d{3})"  # capacitance code (EIA)
    r"([A-Z])"  # tolerance code
    r"(.*)$",  # suffix
)

# Size codes → package string
_SIZE_MAP: dict[str, str] = {
    "155": "0402",
    "185": "0603",
    "21R": "0805",
    "31R": "1206",
    "32R": "1210",
    "55R": "2220",
}

# Voltage codes → rated voltage (V)
_VOLTAGE_MAP: dict[str, float] = {
    "0J": 6.3,
    "1A": 10.0,
    "1C": 16.0,
    "1E": 25.0,
    "1H": 50.0,
    "2A": 100.0,
    "2E": 250.0,
}

# Tolerance codes → fractional tolerance
_TOL_MAP: dict[str, float] = {
    "B": 0.10,
    "C": 0.25,
    "D": 0.50,
    "F": 0.01,
    "G": 0.02,
    "J": 0.05,
    "K": 0.10,
    "M": 0.20,
    "Z": -0.20,  # +80/-20 → treat as 0.20
}


def _decode_eia_cap(code: str) -> float | None:
    """Decode EIA capacitance code (3 digits) to farads.

    E.g. '104' → 100000 pF = 100 nF = 1e-7 F.
    """
    if len(code) != 3 or not code.isdigit():
        return None
    mantissa = int(code[:2])
    exponent = int(code[2])
    return mantissa * 10.0**exponent * 1e-12  # pF → F


def decode_murata_part_number(pn: str) -> dict[str, Any]:
    """Decode Murata part number into metadata dict.

    Used as fallback when CSV columns do not provide the data.
    """
    m = _MURATA_PN_RE.match(pn.strip())
    if m is None:
        return {}
    result: dict[str, Any] = {}
    size_code = m.group(2)
    if size_code in _SIZE_MAP:
        result["package"] = _SIZE_MAP[size_code]
    voltage_code = m.group(5)
    if voltage_code in _VOLTAGE_MAP:
        result["voltage_max_v"] = _VOLTAGE_MAP[voltage_code]
    cap_code = m.group(6)
    cap = _decode_eia_cap(cap_code)
    if cap is not None:
        result["value_nom"] = cap
    tol_code = m.group(7)
    if tol_code in _TOL_MAP:
        result["value_tol_frac"] = abs(_TOL_MAP[tol_code])
    return result


class MurataCSVImporter(GenericCSVImporter):
    """Murata capacitor catalog CSV importer.

    Uses Murata's typical column headers as primary data. Part-number
    decoding is fallback/cross-check only.
    """

    def _column_map(self) -> dict[str, str] | None:
        return {
            "vendor": "Manufacturer",
            "part_number": "Part Number",
            "package": "Size",
            "value_nom": "Capacitance",
            "value_tol_frac": "Tolerance",
            "voltage_max_v": "Rated Voltage",
            "esr_ohm": "ESR",
            "description": "Temperature Characteristic",
        }

    def _preprocess_row(
        self,
        row: dict[str, str],
        resolved_map: dict[str, str | None],
    ) -> dict[str, str]:
        """Inject kind=capacitor and apply PN decoding as fallback."""
        row = dict(row)  # copy

        # Ensure kind is set
        kind_header = resolved_map.get("kind")
        if kind_header is None or not row.get(kind_header, "").strip():
            # Inject a synthetic kind column
            row["__kind__"] = "capacitor"
            resolved_map["kind"] = "__kind__"

        # Inject vendor if not present
        vendor_header = resolved_map.get("vendor")
        if vendor_header is None or not row.get(vendor_header, "").strip():
            row["__vendor__"] = "Murata"
            resolved_map["vendor"] = "__vendor__"

        # Part-number decoding as fallback for missing columns
        pn_header = resolved_map.get("part_number")
        if pn_header:
            pn = row.get(pn_header, "").strip()
            decoded = decode_murata_part_number(pn)
            # Only fill in blanks — CSV columns are primary
            for field, csv_key in [
                ("package", "package"),
                ("voltage_max_v", "voltage_max_v"),
                ("value_nom", "value_nom"),
                ("value_tol_frac", "value_tol_frac"),
            ]:
                header = resolved_map.get(csv_key)
                if header and not row.get(header, "").strip() and field in decoded:
                    row[header] = str(decoded[field])

        return row
