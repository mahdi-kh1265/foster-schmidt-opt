"""Generic column-mapped CSV importer (Prompt 08).

Supports arbitrary CSV layouts via a configurable ``column_map`` that maps
canonical field names to actual CSV headers.  SI-prefix parsing and tolerance
conversion are built in.
"""

from __future__ import annotations

import csv
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from foster_eom.catalog.component import (
    ComponentKind,
    LibraryComponent,
    ModelCondition,
    ModelOrigin,
    ModelTier,
)
from foster_eom.catalog.file_store import compute_sha256
from foster_eom.catalog.importers.base import CatalogImporter, ImportResult

# ---------------------------------------------------------------------------
# SI prefix parsing
# ---------------------------------------------------------------------------

_SI_PREFIXES: dict[str, float] = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "μ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

_VALUE_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*([pnuµμmkMG]?)\s*"
    r"([HhFfΩRr](?:[Hhez]*)?)?\s*$"
)


def parse_si_value(s: str) -> float | None:
    """Parse a numeric value with optional SI prefix.

    Examples: ``'47nH'`` → ``47e-9``, ``'100pF'`` → ``100e-12``,
    ``'2.2'`` → ``2.2``.
    """
    s = s.strip()
    if not s:
        return None
    m = _VALUE_RE.match(s)
    if m is None:
        try:
            return float(s)
        except ValueError:
            return None
    num = float(m.group(1))
    prefix = m.group(2)
    return num * _SI_PREFIXES.get(prefix, 1.0)


def parse_tolerance(s: str) -> float | None:
    """Parse a tolerance string to fractional value.

    Examples: ``'5%'`` → ``0.05``, ``'±2%'`` → ``0.02``, ``'0.1'`` → ``0.1``.
    """
    s = s.strip().replace("±", "").replace("+/-", "")
    if not s:
        return None
    if "%" in s:
        try:
            return float(s.replace("%", "")) / 100.0
        except ValueError:
            return None
    try:
        v = float(s)
        return v if v < 1.0 else v / 100.0  # heuristic: >1 probably percent
    except ValueError:
        return None


def parse_float_opt(s: str) -> float | None:
    """Parse a float, returning None on failure or blank."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return parse_si_value(s)


def infer_kind(s: str) -> ComponentKind | None:
    """Infer component kind from string."""
    s_lower = s.strip().lower()
    for kind in ComponentKind:
        if s_lower == kind.value or s_lower.startswith(kind.value[:3]):
            return kind
    if s_lower in ("cap", "c"):
        return ComponentKind.CAPACITOR
    if s_lower in ("ind", "l"):
        return ComponentKind.INDUCTOR
    if s_lower in ("res", "r"):
        return ComponentKind.RESISTOR
    return None


# ---------------------------------------------------------------------------
# Default column map
# ---------------------------------------------------------------------------

DEFAULT_COLUMN_MAP: dict[str, list[str]] = {
    "kind": ["kind", "type", "component_type"],
    "vendor": ["vendor", "manufacturer", "mfr"],
    "part_number": ["part_number", "mpn", "pn", "part"],
    "package": ["package", "case", "size", "footprint"],
    "description": ["description", "desc"],
    "value_nom": ["value", "value_nom", "nominal"],
    "value_unit": ["unit", "value_unit"],
    "value_tol_frac": ["tolerance", "tol", "value_tol_frac"],
    "voltage_max_v": ["voltage", "voltage_max", "vdc", "rated_voltage"],
    "current_max_a": ["current", "current_max", "irms"],
    "current_sat_a": ["isat", "current_sat", "saturation_current"],
    "srf_hz": ["srf", "srf_hz", "self_resonant_freq"],
    "q_value": ["q", "q_value", "quality_factor"],
    "q_at_f_hz": ["q_freq", "q_at_f_hz", "q_frequency"],
    "esr_ohm": ["esr", "esr_ohm"],
    "dcr_ohm": ["dcr", "dcr_ohm", "dc_resistance"],
}


def _find_column(headers: list[str], aliases: list[str]) -> str | None:
    """Find the first matching header from a list of aliases (case-insensitive)."""
    headers_lower = [h.lower().strip() for h in headers]
    for alias in aliases:
        alias_lower = alias.lower()
        for i, h in enumerate(headers_lower):
            if h == alias_lower:
                return headers[i]
    return None


def _resolve_column_map(
    headers: list[str],
    column_map: dict[str, str] | None,
) -> dict[str, str | None]:
    """Resolve canonical names to actual CSV header names."""
    result: dict[str, str | None] = {}
    for canonical, aliases in DEFAULT_COLUMN_MAP.items():
        if column_map and canonical in column_map:
            result[canonical] = column_map[canonical]
        else:
            result[canonical] = _find_column(headers, aliases)
    return result


# ---------------------------------------------------------------------------
# GenericCSVImporter
# ---------------------------------------------------------------------------


class GenericCSVImporter(CatalogImporter):
    """Generic CSV importer with configurable column mapping.

    Subclasses (vendor adapters) override ``_column_map()`` and optionally
    ``_preprocess_row()`` to handle vendor-specific formats.
    """

    def __init__(self, column_map: dict[str, str] | None = None) -> None:
        self._custom_column_map = column_map

    def _column_map(self) -> dict[str, str] | None:
        """Return custom column mapping. Subclasses override this."""
        return self._custom_column_map

    def _preprocess_row(
        self,
        row: dict[str, str],
        resolved_map: dict[str, str | None],
    ) -> dict[str, str]:
        """Pre-process a row before field extraction. Subclasses override."""
        return row

    def _get_field(
        self,
        row: dict[str, str],
        canonical: str,
        resolved_map: dict[str, str | None],
    ) -> str:
        """Get a field value from a CSV row using the resolved column map."""
        header = resolved_map.get(canonical)
        if header is None:
            return ""
        return row.get(header, "").strip()

    def import_to(
        self,
        library: ComponentLibrary,  # type: ignore[name-defined]  # noqa: F821
        path: Path,
        on_conflict: str = "error",
    ) -> ImportResult:
        """Import components from a CSV file."""

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")

        file_sha = compute_sha256(p)
        now = datetime.now(UTC).isoformat()

        result = ImportResult(import_sha256=file_sha)

        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                result.errors.append("CSV file has no headers.")
                return result

            headers = list(reader.fieldnames)
            resolved = _resolve_column_map(headers, self._column_map())

            for row_num, raw_row in enumerate(reader, start=2):
                row = self._preprocess_row(raw_row, resolved)
                try:
                    comp, model_cond = self._parse_row(
                        row, resolved, file_sha, now, p.name, row_num
                    )
                    cid = library.add(comp, on_conflict=on_conflict)
                    if cid == "__skipped_dup__":
                        result.skipped_dup += 1
                    elif cid == "__skipped_error__":
                        result.skipped_error += 1
                    else:
                        result.inserted += 1
                        if model_cond is not None:
                            model_cond.component_id = cid
                            library.add_model_condition(model_cond)
                except Exception as exc:
                    result.skipped_error += 1
                    result.errors.append(f"Row {row_num}: {exc}")

        return result

    def _parse_row(
        self,
        row: dict[str, str],
        resolved: dict[str, str | None],
        file_sha: str,
        now: str,
        source_name: str,
        row_num: int,
    ) -> tuple[LibraryComponent, ModelCondition | None]:
        """Parse a single CSV row into component + optional model condition."""
        # Required fields
        kind_str = self._get_field(row, "kind", resolved)
        vendor = self._get_field(row, "vendor", resolved)
        part_number = self._get_field(row, "part_number", resolved)
        value_str = self._get_field(row, "value_nom", resolved)

        if not vendor:
            raise ValueError("Missing required field 'vendor'.")
        if not part_number:
            raise ValueError("Missing required field 'part_number'.")
        if not value_str:
            raise ValueError("Missing required field 'value_nom'.")

        # Parse value with unit
        value = parse_si_value(value_str)
        if value is None:
            raise ValueError(f"Cannot parse value '{value_str}'.")

        # Infer kind from explicit field or unit suffix
        kind: ComponentKind | None = None
        if kind_str:
            kind = infer_kind(kind_str)
        if kind is None:
            # Try to infer from unit suffix
            unit = self._get_field(row, "value_unit", resolved).lower()
            if unit in ("h", "nh", "uh", "mh", "ph"):
                kind = ComponentKind.INDUCTOR
            elif unit in ("f", "pf", "nf", "uf", "mf"):
                kind = ComponentKind.CAPACITOR
            elif unit in ("ohm", "r", "ω", "mohm", "kohm"):
                kind = ComponentKind.RESISTOR
        if kind is None:
            raise ValueError("Cannot determine component kind.")

        # Optional fields
        tol = parse_tolerance(self._get_field(row, "value_tol_frac", resolved))
        v_max = parse_float_opt(self._get_field(row, "voltage_max_v", resolved))
        i_max = parse_float_opt(self._get_field(row, "current_max_a", resolved))
        i_sat = parse_float_opt(self._get_field(row, "current_sat_a", resolved))

        comp = LibraryComponent(
            id=str(uuid4()),
            kind=kind,
            vendor=vendor,
            part_number=part_number,
            value_nom=value,
            value_tol_frac=tol,
            voltage_max_v=v_max,
            current_max_a=i_max,
            current_sat_a=i_sat,
            package=self._get_field(row, "package", resolved),
            description=self._get_field(row, "description", resolved),
            import_source=source_name,
            import_sha256=file_sha,
            import_ts=now,
        )
        comp.content_sha256 = comp.compute_content_sha256()

        # Build model condition if parametric data available
        model_cond: ModelCondition | None = None
        esr = parse_float_opt(self._get_field(row, "esr_ohm", resolved))
        dcr = parse_float_opt(self._get_field(row, "dcr_ohm", resolved))
        srf = parse_si_value(self._get_field(row, "srf_hz", resolved))
        q_val = parse_float_opt(self._get_field(row, "q_value", resolved))
        q_freq = parse_si_value(self._get_field(row, "q_at_f_hz", resolved))

        has_parametric = esr is not None or dcr is not None
        if has_parametric:
            params: dict[str, Any] = {}
            if esr is not None:
                params["esr_ohm"] = esr
            if dcr is not None:
                params["dcr_ohm"] = dcr
            # Derive parasitic C from SRF for inductors
            if srf is not None and kind == ComponentKind.INDUCTOR and value > 0:
                import math

                c_par = 1.0 / ((2.0 * math.pi * srf) ** 2 * value)
                params["c_par_f"] = c_par

            model_cond = ModelCondition(
                id=str(uuid4()),
                component_id="",  # filled after insert
                model_tier=ModelTier.PARAMETRIC,
                model_origin=ModelOrigin.VENDOR_PARAMETRIC,
                parametric_params=params,
                srf_hz=srf,
                q_at_f_hz=q_freq,
                q_value=q_val,
                esr_ohm=esr,
                import_ts=now,
            )

        return comp, model_cond
