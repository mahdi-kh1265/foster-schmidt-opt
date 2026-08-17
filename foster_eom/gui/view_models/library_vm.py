"""View models for catalog library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartRow:
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
    n_measured: int
    n_parametric: int
    n_ideal: int
    parts: list[PartRow]
    sha256: str
