"""Controller for EOM measurement import and fitting."""

from __future__ import annotations

import shutil
from pathlib import Path

from foster_eom.measurement.io_csv import load_csv
from foster_eom.measurement.io_s1p import load_s1p


class EomImportCtrl:
    @staticmethod
    def load_measured_file(path: str) -> object:
        """Load .s1p or .csv and return a MeasuredDataset."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".s1p":
            return load_s1p(path)
        elif ext == ".csv":
            return load_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def store_file(source_path: str, project_dir: str) -> str:
        """Copy the file into the project directory and return relative path."""
        src = Path(source_path)
        dst = Path(project_dir) / src.name
        if src.absolute() != dst.absolute():
            shutil.copy2(src, dst)
        return dst.name
