"""Catalog importer base classes (Prompt 08)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary


@dataclass
class ImportResult:
    """Summary of an import operation."""

    inserted: int = 0
    updated: int = 0
    skipped_dup: int = 0  # idempotent re-import (same SHA)
    skipped_error: int = 0
    errors: list[str] = field(default_factory=list)
    import_sha256: str = ""


class CatalogImporter(ABC):
    """Abstract base for catalog importers."""

    @abstractmethod
    def import_to(
        self,
        library: ComponentLibrary,
        path: Path,
        on_conflict: str = "error",
    ) -> ImportResult:
        """Import components from a source file into the library.

        Parameters
        ----------
        library : ComponentLibrary
        path : Path
        on_conflict : str
            ``'error'``, ``'merge'``, or ``'replace'``.
        """
        ...
