"""Murata GJM/GQM S-parameter bulk importer (vendor-specific).

File conventions (observed from gjm-s-v77.zip / gqm-s-v77.zip):
  - .s2p files, series-mode 2-port fixture
  - # Hz S RI R 50 format
  - Filename = full part number (e.g. GJM0225C1C100GB01.s2p)
  - Directory structure: <dielectric_class>/<package>/
  - Freq range: 100 MHz - 30 GHz (401 points)
  - Port conventions: !These Parameters are Measured in Series Mode Connection

Fixture: SERIES topology, port_z=0, port_gnd=1.
Derived component kind: CAPACITOR.
Capacitance decoded from part number (EIA code) as fallback; must be verified
against actual measured SRF / Z data when available.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from foster_eom.catalog.component import (
    ComponentKind,
    LibraryComponent,
    ModelCondition,
    ModelOrigin,
    ModelTier,
)
from foster_eom.catalog.fixture import FixtureSpec, FixtureType
from foster_eom.catalog.importers.base import CatalogImporter, ImportResult

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary

# ---------------------------------------------------------------------------
# Part-number decoder — GJM / GQM
# ---------------------------------------------------------------------------
# GJM0225C1C100GB01 → series=GJM, size=02(25), dielectric=C, voltage=1C, cap=100, tol=G
# GQM22M5C2H1R0BB01 → similar
# GJM0225C1C100GB01:
#   GJM   = series (3)
#   02    = EIA case width code (2 digits)
#   25    = EIA case height code (2 digits)
#   C     = height class (1)
#   1     = temperature class (1)
#   C     = voltage code (1)
#   100   = capacitance EIA (3)
#   G     = tolerance (1)
#   B01   = revision suffix
#
# GQM0335C2D100GB01 -> GQM + 03 + 35 + C + 2 + D + 100 + G + B01
_MURATA_CAP_RE = re.compile(
    r"^(GJM|GQM)"  # series (3)
    r"(\w{4})"  # 4-char size code: digits for GJM (e.g. 0225), alphanum for GQM (e.g. 22M5)
    r"([A-Z])"  # height class
    r"(\d)"  # temperature class
    r"([A-Z])"  # voltage code
    r"([0-9R]{3})"  # capacitance code
    r"([A-Z])"  # tolerance code
    r"(.*)$"  # suffix
)

_VOLTAGE_MAP: dict[str, float] = {
    "J": 6.3,
    "A": 10.0,
    "C": 16.0,
    "E": 25.0,
    "H": 50.0,
    "D": 100.0,
    "G": 4.0,
    "B": 12.5,
    "F": 3.15,
    "R": 1.25,
    "S": 1.5,
    "T": 1.8,
    "U": 2.0,
}

_TOL_MAP: dict[str, float] = {
    "A": 0.005,
    "B": 0.10,
    "C": 0.25,
    "D": 0.50,
    "F": 0.01,
    "G": 0.02,
    "J": 0.05,
    "K": 0.10,
    "M": 0.20,
    "W": 0.05,
    "Z": 0.20,
}


def _decode_eia_cap(code: str) -> float | None:
    """Decode 3-char EIA code (e.g. '100'→10e-12, '1R0'→1e-12) to farads."""
    if "R" in code:
        # Sub-pF: '1R0'→1.0pF, '2R2'→2.2pF
        try:
            val = float(code.replace("R", ".")) * 1e-12
            return val
        except ValueError:
            return None
    if len(code) == 3 and code.isdigit():
        mantissa = int(code[:2])
        exp = int(code[2])
        return mantissa * 10.0**exp * 1e-12
    return None


def decode_murata_gjm_gqm(pn: str) -> dict:
    """Decode Murata GJM/GQM part number into metadata dict."""
    m = _MURATA_CAP_RE.match(pn.strip())
    if m is None:
        return {}
    series = m.group(1)
    voltage_code = m.group(5)  # single-char voltage code
    cap_code = m.group(6)
    tol_code = m.group(7)

    result: dict = {"vendor": "Murata", "series": series}
    cap = _decode_eia_cap(cap_code)
    if cap is not None:
        result["value_nom"] = cap
    # Voltage: single-char map (digit removed from index)
    if voltage_code in _VOLTAGE_MAP:
        result["voltage_max_v"] = _VOLTAGE_MAP[voltage_code]
    if tol_code in _TOL_MAP:
        result["value_tol_frac"] = _TOL_MAP[tol_code]
    return result


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

_SERIES_FIXTURE = FixtureSpec(fixture_type=FixtureType.SERIES, port_z=0, port_gnd=1)


class MurataGJMGQMImporter(CatalogImporter):
    """Import Murata GJM/GQM .s2p S-parameter packs.

    Accepts a single .s2p file path OR a directory (recurses for *.s2p).
    Creates/upserts a LibraryComponent per part and attaches a MEASURED
    model_condition with SERIES fixture semantics.
    """

    def __init__(self, *, vendor: str = "Murata", max_files: int | None = None) -> None:
        self.vendor = vendor
        self.max_files = max_files

    def import_to(
        self,
        library: ComponentLibrary,
        path: Path,
        on_conflict: str = "error",
    ) -> ImportResult:
        import skrf  # type: ignore[import-untyped]

        result = ImportResult()
        now = datetime.now(UTC).isoformat()

        files = [path] if path.is_file() else sorted(path.rglob("*.s2p"))[: self.max_files]

        for s2p_file in files:
            try:
                self._import_one(library, s2p_file, now, result, on_conflict, skrf)
            except Exception as exc:
                result.skipped_error += 1
                result.errors.append(f"{s2p_file.name}: {exc}")

        return result

    def _import_one(
        self,
        library: ComponentLibrary,
        s2p_file: Path,
        now: str,
        result: ImportResult,
        on_conflict: str,
        skrf: object,
    ) -> None:
        pn = s2p_file.stem  # full part number from filename

        # Decode metadata from part number
        meta = decode_murata_gjm_gqm(pn)
        value_nom = meta.get("value_nom")
        if value_nom is None:
            result.skipped_error += 1
            result.errors.append(f"{s2p_file.name}: cannot decode capacitance from PN '{pn}'")
            return

        # Parse S2P to get frequency range
        ntwk = skrf.Network(str(s2p_file))  # type: ignore[attr-defined]
        f_hz = np.asarray(ntwk.f, dtype=np.float64)
        if len(f_hz) < 2:
            result.skipped_error += 1
            result.errors.append(f"{s2p_file.name}: too few frequency points")
            return

        # Build/upsert component
        comp = LibraryComponent(
            id=str(uuid4()),
            vendor=self.vendor,
            part_number=pn,
            kind=ComponentKind.CAPACITOR,
            value_nom=float(value_nom),
            value_tol_frac=meta.get("value_tol_frac", 0.10),
            voltage_max_v=meta.get("voltage_max_v"),
            description=f"Murata {meta.get('series', '')} MLCC",
            package="",
            import_source="vendor_s2p",
            import_sha256="",
            import_ts=now,
            content_sha256="",
            user_notes="",
        )
        comp_id_result = library.add(comp, on_conflict=on_conflict)
        if comp_id_result not in ("__skipped_dup__",) and not comp_id_result:
            result.skipped_error += 1
            return
        if comp_id_result == "__skipped_dup__":
            result.skipped_dup += 1
        else:
            result.inserted += 1

        # Re-fetch to get actual comp.id
        try:
            stored_comp = library.get_by_part(self.vendor, pn)
        except KeyError:
            result.skipped_error += 1
            result.errors.append(f"{s2p_file.name}: comp lookup failed after insert")
            return

        # Store file and create model condition
        sha = library.file_store.store(s2p_file)
        mc = ModelCondition(
            id=str(uuid4()),
            component_id=stored_comp.id,
            model_tier=ModelTier.MEASURED,
            model_origin=ModelOrigin.VENDOR_TOUCHSTONE,
            model_file_sha256=sha,
            model_file_ext=".s2p",
            n_ports=2,
            validity_hz_lo=float(f_hz[0]),
            validity_hz_hi=float(f_hz[-1]),
            import_ts=now,
            fixture_type=_SERIES_FIXTURE.fixture_type.value,
            fixture_port_z=_SERIES_FIXTURE.port_z,
            fixture_port_gnd=_SERIES_FIXTURE.port_gnd,
        )
        library.add_model_condition(mc)
