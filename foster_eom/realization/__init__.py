"""Discrete catalog realization package (Prompt 09).

Public API
----------
realize()             : top-level orchestrator
RealizationSpec       : configuration
RealizationBudget     : MNA-solve budget
SlotSpec              : per-slot eligibility
NeighborhoodEntry     : frozen catalog candidate
CatalogCombo          : one evaluated realization
RealizationResult     : complete result
RealizationDiagnostics: search accounting
build_slot_specs()    : auto-build slot specs from context + branch coords
build_neighborhoods() : query catalog per slot
build_substituted_graph(): substitute catalog models into a CircuitGraph
"""

from __future__ import annotations

from foster_eom.realization.beam import generate_combos
from foster_eom.realization.neighborhoods import build_neighborhoods, build_slot_specs
from foster_eom.realization.result import (
    CatalogCombo,
    RealizationDiagnostics,
    RealizationResult,
)
from foster_eom.realization.runner import realize
from foster_eom.realization.spec import (
    NeighborhoodEntry,
    RealizationBudget,
    RealizationSpec,
    SlotSpec,
)
from foster_eom.realization.substitute import build_substituted_graph

__all__ = [
    "CatalogCombo",
    "NeighborhoodEntry",
    "RealizationBudget",
    "RealizationDiagnostics",
    "RealizationResult",
    "RealizationSpec",
    "SlotSpec",
    "build_neighborhoods",
    "build_slot_specs",
    "build_substituted_graph",
    "generate_combos",
    "realize",
]
