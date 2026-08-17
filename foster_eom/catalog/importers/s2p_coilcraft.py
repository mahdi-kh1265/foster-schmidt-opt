"""Coilcraft ADS RF Library .s2p bulk importer.

File conventions (observed from coilcraft_ads_rf_library.zip):
  - .s2p files in circuit/data/ subdirectory
  - # MHz S MA R 50 format (magnitude-angle)
  - Filename = '<Series>-<Value>.s2p'
    e.g. '016008C-10N.s2p' -> series='016008C', value code='10N' -> 10 nH
         '0201AF-111.s2p'  -> series='0201AF', value code='111' (EIA: 110 nH)
         '0402CS-8N2.s2p'  -> 8.2 nH
  - Freq range: ~1 MHz to several GHz (log-spaced)
  - Fixture: 2-port SERIES (port1 = input, port2 = output, DUT in-line)

Value code conventions (Coilcraft):
  - 'XNY' format: X.Y nH (e.g. '1N2' = 1.2 nH, '10N' = 10 nH, 'N45' = 0.45 nH)
  - EIA 3-digit (e.g. '111' = 11 * 10^1 nH = 110 nH)
"""

from __future__ import annotations

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

_SERIES_FIXTURE = FixtureSpec(fixture_type=FixtureType.SERIES, port_z=0, port_gnd=1)

# Package → size string
_PACKAGE_MAP: dict[str, str] = {
    "016008C": "01006",
    "0201AF": "0201",
    "0201DS": "0201",
    "0201HL": "0201",
    "026011C": "0201",
    "026011F": "0201",
    "0302CS": "0302",
    "0402CS": "0402",
    "0402CT": "0402",
    "0402DC": "0402",
    "0402FL": "0402",
    "0402HL": "0402",
    "0402HP": "0402",
    "0402PA": "0402",
    "0403HQ": "0403",
    "0603AF": "0603",
    "0603CS": "0603",
    "0603CT": "0603",
    "0603DC": "0603",
    "0603HC": "0603",
    "0603HL": "0603",
    "0603HP": "0603",
    "0603LS": "0603",
    "0604HQ": "0604",
    "0805AF": "0805",
    "0805CS": "0805",
    "0805HQ": "0805",
    "1008AF": "1008",
    "1008CS": "1008",
    "1008HQ": "1008",
}


def decode_coilcraft_value(code: str) -> float | None:
    """Decode Coilcraft value code to henries.

    Formats:
    - 'RXY'  : 0.XY nH sub-1nH (e.g. 'R10' -> 0.10 nH, 'R45' -> 0.45 nH)
    - 'XNY'  : X.Y nH (e.g. '1N2' -> 1.2 nH, '10N' -> 10 nH, 'N45' -> 0.45 nH)
    - '###'  : EIA 3-digit nH (e.g. '111' -> 110 nH, '100' -> 10 nH)
    """
    code = code.upper().strip()

    # R-prefix: sub-1nH, e.g. 'R10' -> 0.10 nH
    if code.startswith("R") and len(code) == 3 and code[1:].isdigit():
        try:
            return float(f"0.{code[1:]}") * 1e-9
        except ValueError:
            return None

    # XNY format: sub-10nH and nH range
    if "N" in code:
        parts = code.split("N")
        if len(parts) == 2:
            int_part = parts[0] if parts[0] else "0"
            frac_part = parts[1] if parts[1] else "0"
            try:
                val_nh = float(f"{int_part}.{frac_part}")
                return val_nh * 1e-9
            except ValueError:
                return None

    # EIA 3-digit
    if len(code) == 3 and code.isdigit():
        mantissa = int(code[:2])
        exp = int(code[2])
        return mantissa * 10.0**exp * 1e-9  # nH

    return None


def decode_coilcraft_filename(stem: str) -> tuple[str, str, float | None]:
    """Parse 'SERIES-VALUECODE' → (series, part_number, value_h).

    Some filenames use non-standard codes (e.g. 'WA3096-AL') — these will
    have value_h=None and will be skipped.
    """
    parts = stem.split("-", 1)
    if len(parts) != 2:
        return stem, stem, None
    series, val_code = parts
    value_h = decode_coilcraft_value(val_code)
    return series, stem, value_h


class CoilcraftS2PImporter(CatalogImporter):
    """Import Coilcraft ADS RF Library .s2p files.

    Accepts a directory (recurses for *.s2p) or a single .s2p file.
    Creates/upserts a LibraryComponent per part and attaches a MEASURED
    model_condition with SERIES fixture semantics.
    """

    def __init__(self, *, vendor: str = "Coilcraft", max_files: int | None = None) -> None:
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
        series, pn, value_h = decode_coilcraft_filename(s2p_file.stem)

        if value_h is None:
            result.skipped_error += 1
            result.errors.append(
                f"{s2p_file.name}: cannot decode inductance value from '{s2p_file.stem}'"
            )
            return

        # Parse S2P
        ntwk = skrf.Network(str(s2p_file))  # type: ignore[attr-defined]
        f_hz = np.asarray(ntwk.f, dtype=np.float64)
        if len(f_hz) < 2:
            result.skipped_error += 1
            result.errors.append(f"{s2p_file.name}: too few frequency points")
            return

        package = _PACKAGE_MAP.get(series, series)

        # Build/upsert component
        comp = LibraryComponent(
            id=str(uuid4()),
            vendor=self.vendor,
            part_number=pn,
            kind=ComponentKind.INDUCTOR,
            value_nom=float(value_h),
            value_tol_frac=0.05,
            package=package,
            description=f"Coilcraft {series} RF inductor",
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

        try:
            stored_comp = library.get_by_part(self.vendor, pn)
        except KeyError:
            result.skipped_error += 1
            result.errors.append(f"{s2p_file.name}: comp lookup failed after insert")
            return

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
