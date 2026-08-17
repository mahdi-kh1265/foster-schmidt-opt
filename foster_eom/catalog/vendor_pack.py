"""Vendor-pack import workflow for already-downloaded manufacturer archives (Prompt 09).

Accepts locally-present vendor ZIP archives or directories and imports them
into a P08 ComponentLibrary with full provenance tracking.  No internet
access required.  Vendor files must **not** be committed to the repository.

Typical usage
-------------
::

    from foster_eom.catalog.vendor_pack import VendorPackWorkflow, VendorPackSpec

    spec = VendorPackSpec(
        vendor="Coilcraft",
        adapter="coilcraft_csv",
        source_path=Path("/local/downloads/coilcraft_xal_xfl.zip"),
    )
    wf = VendorPackWorkflow(library)
    manifest = wf.run(spec)
    print(manifest.summary())

Provenance fields recorded per run
-----------------------------------
* vendor / adapter / adapter_version
* source path (not committed) + source SHA-256
* import timestamp (UTC ISO-8601)
* per-file import counts (inserted / skipped_dup / errors)
* resulting library SHA-256 (snapshot after import)
* manifest written to ``<library_dir>/import_manifests/<timestamp>_<vendor>.json``

The manifest JSON is safe to commit; the vendor files themselves must not be.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foster_eom.catalog.importers.base import ImportResult

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary

# ---------------------------------------------------------------------------
# Adapter version registry
# ---------------------------------------------------------------------------

#: Bump these when parsing/mapping logic changes to preserve provenance.
_ADAPTER_VERSIONS: dict[str, str] = {
    "coilcraft_csv": "1.0.0",
    "murata_csv": "1.0.0",
    "generic_csv": "1.0.0",
    "touchstone": "1.0.0",
    "s2p_murata_gjm_gqm": "1.0.0",
    "s2p_coilcraft": "1.0.0",
}

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class VendorPackSpec:
    """Specification for one vendor-pack import run.

    Parameters
    ----------
    vendor : str
        Display name preserved in provenance (e.g. ``"Coilcraft"``).
    adapter : str
        Adapter key: ``"coilcraft_csv"``, ``"murata_csv"``, ``"generic_csv"``,
        or ``"touchstone"``.
    source_path : Path
        Absolute path to the local ZIP archive or directory.
        Must exist; must **not** be in the repository.
    measurement_plane : str
        Intended measurement plane description (provenance only).
    glob_pattern : str
        Glob pattern relative to the (extracted) source to find files.
        Default ``"**/*.csv"`` for CSV adapters.
    on_conflict : str
        ``"error"`` (default), ``"merge"``, or ``"replace"``.
    extra_meta : dict[str, Any]
        Arbitrary extra provenance fields written into the manifest.
    """

    vendor: str
    adapter: str
    source_path: Path
    measurement_plane: str = "vendor_datasheet"
    glob_pattern: str = ""
    on_conflict: str = "error"
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        if not self.glob_pattern:
            self.glob_pattern = "**/*.s1p" if self.adapter == "touchstone" else "**/*.csv"

    @property
    def adapter_version(self) -> str:
        return _ADAPTER_VERSIONS.get(self.adapter, "unknown")


# ---------------------------------------------------------------------------
# Per-file result
# ---------------------------------------------------------------------------


@dataclass
class FileImportRecord:
    """Import result for a single file."""

    filename: str
    sha256: str
    inserted: int
    updated: int
    skipped_dup: int
    skipped_error: int
    errors: list[str]
    import_sha256: str  # from ImportResult


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class VendorPackManifest:
    """Full provenance manifest for one vendor-pack import run.

    This document is safe to commit; it contains no raw component data.
    """

    run_id: str  # UUID-like timestamp key
    vendor: str
    adapter: str
    adapter_version: str
    measurement_plane: str
    source_path: str  # absolute; do NOT commit vendor files
    source_sha256: str
    import_timestamp_utc: str
    on_conflict: str
    extra_meta: dict[str, Any]

    # Per-file accounting
    file_records: list[FileImportRecord]

    # Aggregate
    n_files_processed: int
    n_inserted_total: int
    n_updated_total: int
    n_skipped_dup_total: int
    n_error_total: int
    all_errors: list[str]

    # Library state after import
    library_sha256: str  # SHA-256 of db file after all imports

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary."""
        lines = [
            "=== Vendor Pack Import Manifest ===",
            f"Vendor          : {self.vendor}",
            f"Adapter         : {self.adapter} v{self.adapter_version}",
            f"Measurement plane: {self.measurement_plane}",
            f"Source          : {self.source_path}",
            f"Source SHA-256  : {self.source_sha256[:16]}…",
            f"Timestamp (UTC) : {self.import_timestamp_utc}",
            f"Files processed : {self.n_files_processed}",
            f"Inserted        : {self.n_inserted_total}",
            f"Updated         : {self.n_updated_total}",
            f"Skipped (dup)   : {self.n_skipped_dup_total}",
            f"Errors          : {self.n_error_total}",
            f"Library SHA-256 : {self.library_sha256[:16]}…",
        ]
        if self.all_errors:
            lines.append("First error     : " + self.all_errors[0])
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON."""
        d = asdict(self)
        return json.dumps(d, indent=indent, default=str)

    @classmethod
    def from_json(cls, text: str) -> VendorPackManifest:
        """Deserialize manifest from JSON."""
        d = json.loads(text)
        d["file_records"] = [FileImportRecord(**r) for r in d.get("file_records", [])]
        return cls(**d)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class VendorPackWorkflow:
    """Orchestrates a vendor-pack import into the P08 ComponentLibrary.

    Parameters
    ----------
    library : ComponentLibrary
        Open library to import into.
    manifest_dir : Path | None
        Directory to write manifest JSON files.  Defaults to
        ``<library.db_path.parent>/import_manifests/``.
    """

    def __init__(
        self,
        library: ComponentLibrary,
        manifest_dir: Path | None = None,
    ) -> None:
        self.library = library
        self.manifest_dir = manifest_dir or (library.db_path.parent / "import_manifests")
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, spec: VendorPackSpec) -> VendorPackManifest:
        """Execute one vendor-pack import and return the provenance manifest.

        Parameters
        ----------
        spec : VendorPackSpec

        Returns
        -------
        VendorPackManifest
        """
        if not spec.source_path.exists():
            raise FileNotFoundError(
                f"Vendor pack source not found: {spec.source_path}\n"
                "Download manufacturer data files first; do not commit them."
            )

        ts = datetime.datetime.now(datetime.UTC)
        ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{ts_str}_{spec.vendor.lower().replace(' ', '_')}"

        source_sha = _sha256_path(spec.source_path)

        # Extract ZIP or use directory directly
        with _source_context(spec.source_path) as work_dir:
            files = sorted(work_dir.glob(spec.glob_pattern))
            importer = _make_importer(spec.adapter)

            file_records: list[FileImportRecord] = []
            for src_file in files:
                rec = self._import_file(src_file, spec, importer)
                file_records.append(rec)

        lib_sha = _sha256_path(self.library.db_path)

        n_ins = sum(r.inserted for r in file_records)
        n_upd = sum(r.updated for r in file_records)
        n_dup = sum(r.skipped_dup for r in file_records)
        n_err = sum(r.skipped_error for r in file_records)
        all_err = [e for r in file_records for e in r.errors]

        manifest = VendorPackManifest(
            run_id=run_id,
            vendor=spec.vendor,
            adapter=spec.adapter,
            adapter_version=spec.adapter_version,
            measurement_plane=spec.measurement_plane,
            source_path=str(spec.source_path.resolve()),
            source_sha256=source_sha,
            import_timestamp_utc=ts.isoformat(),
            on_conflict=spec.on_conflict,
            extra_meta=spec.extra_meta,
            file_records=file_records,
            n_files_processed=len(file_records),
            n_inserted_total=n_ins,
            n_updated_total=n_upd,
            n_skipped_dup_total=n_dup,
            n_error_total=n_err,
            all_errors=all_err,
            library_sha256=lib_sha,
        )

        # Write manifest
        manifest_path = self.manifest_dir / f"{run_id}.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")

        return manifest

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _import_file(
        self,
        src_file: Path,
        spec: VendorPackSpec,
        importer: Any,
    ) -> FileImportRecord:
        file_sha = _sha256_path(src_file)
        try:
            result: ImportResult = importer.import_to(
                self.library,
                src_file,
                on_conflict=spec.on_conflict,
            )
        except Exception as exc:
            return FileImportRecord(
                filename=src_file.name,
                sha256=file_sha,
                inserted=0,
                updated=0,
                skipped_dup=0,
                skipped_error=1,
                errors=[str(exc)],
                import_sha256="",
            )

        return FileImportRecord(
            filename=src_file.name,
            sha256=file_sha,
            inserted=result.inserted,
            updated=result.updated,
            skipped_dup=result.skipped_dup,
            skipped_error=result.skipped_error,
            errors=result.errors,
            import_sha256=result.import_sha256,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_path(path: Path) -> str:
    """Return hex SHA-256 of a file (or directory tree, sorted)."""
    if path.is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    # Directory: hash sorted file contents
    h = hashlib.sha256()
    for fp in sorted(path.rglob("*")):
        if fp.is_file():
            h.update(fp.relative_to(path).as_posix().encode())
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
    return h.hexdigest()


class _source_context:
    """Context manager: unzip to tmpdir if ZIP, else use directory as-is."""

    def __init__(self, source_path: Path) -> None:
        self._source = source_path
        self._tmpdir: str | None = None

    def __enter__(self) -> Path:
        if self._source.is_file() and self._source.suffix.lower() == ".zip":
            self._tmpdir = tempfile.mkdtemp(prefix="fseom_vendorpack_")
            with zipfile.ZipFile(self._source) as zf:
                zf.extractall(self._tmpdir)
            return Path(self._tmpdir)
        elif self._source.is_dir():
            return self._source
        else:
            raise ValueError(f"source_path must be a .zip file or directory, got: {self._source}")

    def __exit__(self, *_: Any) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


def _make_importer(adapter: str) -> Any:
    """Instantiate the correct importer for the given adapter key."""
    match adapter:
        case "coilcraft_csv":
            from foster_eom.catalog.importers.csv_coilcraft import CoilcraftCSVImporter

            return CoilcraftCSVImporter()
        case "murata_csv":
            from foster_eom.catalog.importers.csv_murata import MurataCSVImporter

            return MurataCSVImporter()
        case "generic_csv":
            from foster_eom.catalog.importers.csv_generic import GenericCSVImporter

            return GenericCSVImporter()
        case "touchstone":
            from foster_eom.catalog.importers.touchstone import TouchstoneImporter

            return TouchstoneImporter()
        case "s2p_murata_gjm_gqm":
            from foster_eom.catalog.importers.s2p_murata import MurataGJMGQMImporter

            return MurataGJMGQMImporter()
        case "s2p_coilcraft":
            from foster_eom.catalog.importers.s2p_coilcraft import CoilcraftS2PImporter

            return CoilcraftS2PImporter()
        case _:
            raise ValueError(
                f"Unknown adapter {adapter!r}. Valid adapters: {sorted(_ADAPTER_VERSIONS)}"
            )
