"""Touchstone bulk importer (Prompt 08).

Imports .s1p/.s2p/.sNp files from a directory into the content-addressed
file store and creates model_conditions records with tier='measured'.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from foster_eom.catalog.component import (
    ModelCondition,
    ModelOrigin,
    ModelTier,
)
from foster_eom.catalog.fixture import FixtureSpec
from foster_eom.catalog.importers.base import CatalogImporter, ImportResult

# Match Touchstone file extensions: .s1p, .s2p, ..., .s99p
_TOUCHSTONE_RE = re.compile(r"\.s(\d+)p$", re.IGNORECASE)


def _default_filename_to_part(filename: str) -> str:
    """Default: strip Touchstone extension to get part number."""
    return _TOUCHSTONE_RE.sub("", filename)


class TouchstoneImporter(CatalogImporter):
    """Bulk-import Touchstone files from a directory.

    Each file is stored in the content-addressed file store and a
    ``model_conditions`` record is created with ``model_tier='measured'``.
    """

    def __init__(
        self,
        *,
        fixture: FixtureSpec | None = None,
        filename_to_part: Callable[[str], str] | None = None,
        vendor: str = "",
    ) -> None:
        self.fixture = fixture
        self.filename_to_part = filename_to_part or _default_filename_to_part
        self.vendor = vendor

    def import_to(
        self,
        library: ComponentLibrary,  # type: ignore[name-defined]  # noqa: F821
        path: Path,
        on_conflict: str = "error",
    ) -> ImportResult:
        """Import all Touchstone files from a directory."""

        directory = Path(path)
        if not directory.is_dir():
            raise ValueError(f"Expected a directory, got: {directory}")

        result = ImportResult()
        now = datetime.now(UTC).isoformat()

        files = sorted(
            f for f in directory.iterdir() if f.is_file() and _TOUCHSTONE_RE.search(f.name)
        )

        for ts_file in files:
            try:
                self._import_one(library, ts_file, now, result, on_conflict)
            except Exception as exc:
                result.skipped_error += 1
                result.errors.append(f"{ts_file.name}: {exc}")

        return result

    def _import_one(
        self,
        library: ComponentLibrary,  # type: ignore[name-defined]  # noqa: F821
        ts_file: Path,
        now: str,
        result: ImportResult,
        on_conflict: str,
    ) -> None:
        import skrf  # type: ignore[import-untyped]

        # Parse to get port count and validity range
        ntwk = skrf.Network(str(ts_file))
        n_ports = ntwk.nports
        f_hz = np.asarray(ntwk.f, dtype=np.float64)

        # Multiport requires fixture
        if n_ports > 1 and self.fixture is None:
            raise ValueError(f"FixtureSpec required for {n_ports}-port file '{ts_file.name}'.")

        # Store in content-addressed file store
        sha = library.file_store.store(ts_file)

        # Determine part number
        part_number = self.filename_to_part(ts_file.name)

        # Find the component
        vendor = self.vendor
        try:
            comp = library.get_by_part(vendor, part_number)
        except KeyError:
            result.skipped_error += 1
            result.errors.append(
                f"{ts_file.name}: No component found for vendor='{vendor}', part='{part_number}'."
            )
            return

        # Create model condition
        ext = ts_file.suffix.lower()
        mc = ModelCondition(
            id=str(uuid4()),
            component_id=comp.id,
            model_tier=ModelTier.MEASURED,
            model_origin=ModelOrigin.VENDOR_TOUCHSTONE,
            model_file_sha256=sha,
            model_file_ext=ext,
            n_ports=n_ports,
            validity_hz_lo=float(f_hz[0]),
            validity_hz_hi=float(f_hz[-1]),
            import_ts=now,
        )

        if self.fixture is not None and n_ports > 1:
            mc.fixture_type = self.fixture.fixture_type.value
            mc.fixture_port_z = self.fixture.port_z
            mc.fixture_port_gnd = self.fixture.port_gnd

        library.add_model_condition(mc)
        result.inserted += 1
