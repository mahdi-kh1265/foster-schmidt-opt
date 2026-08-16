"""Result and candidate data structures (spec §6.9).

These are skeleton structures for Prompt 01.  The full result-population
and serialization logic will be built in later milestones as the circuit
engine, optimizer, and analysis modules come online.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    """Result data for a single optimizer candidate (spec §6.9).

    Mutable during optimization; frozen after finalization.

    Attributes
    ----------
    candidate_id : str
        Unique identifier.
    topology_id : str
        Topology identifier.
    continuous_variables : dict[str, float]
        Optimized continuous variable values.
    resolved_values : dict[str, float]
        Resolved L/C/R values in SI units.
    catalog_parts : dict[str, Any]
        Mapping from logical element to catalog part info.
    pole_locations_hz : list[float]
        Pole frequencies in Hz.
    objective_terms : dict[str, float]
        Individual objective term values.
    constraint_margins : dict[str, float]
        Normalized constraint margins (negative = violated).
    warnings : list[dict[str, Any]]
        Structured warnings from evaluation.
    solver_diagnostics : dict[str, Any]
        Solver-specific diagnostic data.
    feasible : bool
        Whether all hard constraints are satisfied.
    """

    candidate_id: str = ""
    topology_id: str = ""
    continuous_variables: dict[str, float] = Field(default_factory=dict)
    resolved_values: dict[str, float] = Field(default_factory=dict)
    catalog_parts: dict[str, Any] = Field(default_factory=dict)
    pole_locations_hz: list[float] = Field(default_factory=list)
    objective_terms: dict[str, float] = Field(default_factory=dict)
    constraint_margins: dict[str, float] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    solver_diagnostics: dict[str, Any] = Field(default_factory=dict)
    feasible: bool = False
