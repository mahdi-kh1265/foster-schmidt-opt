"""Measured EOM characterization backend (Prompt 07).

This package provides:

- ``MeasuredDataset`` — immutable container for measured S11/impedance data
- ``MeasuredOnePortModel`` — tabulated model for MNA evaluation
- ``load_s1p`` / ``load_csv`` — data import from Touchstone and CSV
- ``fit_lossy_cap`` / ``fit_mbvd`` — equivalent-circuit fitting
- ``FitResult`` / ``FitDiagnostics`` — fit outputs and diagnostics
"""

from foster_eom.measurement.dataset import MeasuredDataset, SourceQuantity
from foster_eom.measurement.fitting import (
    FitDiagnostics,
    FitDomain,
    FitResult,
    fit_lossy_cap,
    fit_mbvd,
)
from foster_eom.measurement.io_csv import load_csv
from foster_eom.measurement.io_s1p import load_s1p
from foster_eom.measurement.measured_model import MeasuredOnePortModel

__all__ = [
    "FitDiagnostics",
    "FitDomain",
    "FitResult",
    "MeasuredDataset",
    "MeasuredOnePortModel",
    "SourceQuantity",
    "fit_lossy_cap",
    "fit_mbvd",
    "load_csv",
    "load_s1p",
]
