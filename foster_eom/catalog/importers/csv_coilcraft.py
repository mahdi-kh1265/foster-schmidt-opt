"""Coilcraft CSV adapter (Prompt 08).

Genuine parsing for Coilcraft XAL/XFL/LPS series inductor catalogs.
Derives parasitic capacitance from SRF when available.
CSV columns are primary data; part-number decoding is fallback only.
"""

from __future__ import annotations

import re
from typing import Any

from foster_eom.catalog.importers.csv_generic import GenericCSVImporter

# Coilcraft part-number decoder (e.g. XAL5030-472MED)
_COILCRAFT_PN_RE = re.compile(
    r"^(XAL|XFL|LPS|XEL|XGL|SLC|MSD)"  # series
    r"(\d{4})"  # size code (e.g. 5030 -> 5.0mm x 3.0mm)
    r"[-]?"
    r"(\d{3})"  # inductance code (EIA)
    r"([A-Z]?)"  # tolerance code
    r"(.*)$",
)

_COILCRAFT_TOL_MAP: dict[str, float] = {
    "F": 0.01,
    "G": 0.02,
    "J": 0.05,
    "K": 0.10,
    "M": 0.20,
    "L": 0.15,
}


def _decode_eia_inductance(code: str) -> float | None:
    """Decode EIA inductance code (3 digits) to henries.

    E.g. '472' → 4700 µH = 4.7 mH = 4.7e-3 H.
    Wait — Coilcraft uses µH convention:
    '472' -> 4.7 uH = 4.7e-6 H (first two digits x 10^third digit, in uH).
    """
    if len(code) != 3 or not code.isdigit():
        return None
    mantissa = int(code[:2])
    exponent = int(code[2])
    return mantissa * 10.0**exponent * 1e-6  # µH → H


def decode_coilcraft_part_number(pn: str) -> dict[str, Any]:
    """Decode Coilcraft part number into metadata dict."""
    m = _COILCRAFT_PN_RE.match(pn.strip())
    if m is None:
        return {}
    result: dict[str, Any] = {}
    series = m.group(1)
    size = m.group(2)
    result["package"] = f"{series}{size}"

    ind_code = m.group(3)
    ind = _decode_eia_inductance(ind_code)
    if ind is not None:
        result["value_nom"] = ind

    tol_code = m.group(4)
    if tol_code in _COILCRAFT_TOL_MAP:
        result["value_tol_frac"] = _COILCRAFT_TOL_MAP[tol_code]

    return result


class CoilcraftCSVImporter(GenericCSVImporter):
    """Coilcraft inductor catalog CSV importer.

    Uses Coilcraft's typical column headers as primary data.
    Derives parasitic capacitance from SRF: C_par = 1/((2π·SRF)²·L).
    """

    def _column_map(self) -> dict[str, str] | None:
        return {
            "vendor": "Manufacturer",
            "part_number": "Part Number",
            "package": "Size",
            "value_nom": "Inductance",
            "value_tol_frac": "Tolerance",
            "current_max_a": "Irms",
            "current_sat_a": "Isat",
            "dcr_ohm": "DCR Typ",
            "srf_hz": "SRF Min",
            "description": "Series",
        }

    def _preprocess_row(
        self,
        row: dict[str, str],
        resolved_map: dict[str, str | None],
    ) -> dict[str, str]:
        """Inject kind=inductor and apply PN decoding as fallback."""
        row = dict(row)

        # Ensure kind is set
        kind_header = resolved_map.get("kind")
        if kind_header is None or not row.get(kind_header, "").strip():
            row["__kind__"] = "inductor"
            resolved_map["kind"] = "__kind__"

        # Inject vendor if not present
        vendor_header = resolved_map.get("vendor")
        if vendor_header is None or not row.get(vendor_header, "").strip():
            row["__vendor__"] = "Coilcraft"
            resolved_map["vendor"] = "__vendor__"

        # Part-number decoding as fallback
        pn_header = resolved_map.get("part_number")
        if pn_header:
            pn = row.get(pn_header, "").strip()
            decoded = decode_coilcraft_part_number(pn)
            for field, csv_key in [
                ("package", "package"),
                ("value_nom", "value_nom"),
                ("value_tol_frac", "value_tol_frac"),
            ]:
                header = resolved_map.get(csv_key)
                if header and not row.get(header, "").strip() and field in decoded:
                    row[header] = str(decoded[field])

        return row
