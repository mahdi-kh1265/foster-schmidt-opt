"""View models for catalog library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartRow:
    id: str
    vendor: str
    part_number: str
    kind: str
    value: str
    tier: str
    validity: str
    origin: str


@dataclass(frozen=True)
class LibraryStats:
    total_parts: int
    n_inductors: int
    n_capacitors: int
    n_measured: int
    n_parametric: int
    n_ideal: int
    parts: list[PartRow]
    sha256: str


@dataclass(frozen=True)
class ComponentDetailsVM:
    vendor: str
    part_number: str
    kind: str
    package: str
    value_nom: str
    tolerance: str
    dcr_esr: str
    srf: str
    voltage_rating: str
    current_rating: str
    validity_hz: str
    model_tier: str
    model_origin: str
    model_file_sha256: str
    is_synthetic: bool
