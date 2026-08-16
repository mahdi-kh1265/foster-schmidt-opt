"""Result and candidate data structures (spec §6.9).

Prompt 01 established the skeleton.  Prompt 05 fills in the full contract:
domain identity, Foster coefficients, feasibility metrics, typed circuit
solution summaries, and complete solver provenance.

All mutable container fields use ``Field(default_factory=...)``; no shared
mutable defaults exist.  Rank is conveyed by tuple position in
``OptimizationResult.candidates``, not by a field on this model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Typed sub-models for circuit solution summaries
# ---------------------------------------------------------------------------


class TargetSolutionSummary(BaseModel, frozen=True):
    """Per-target-frequency circuit solution summary.

    Attributes
    ----------
    frequency_hz : float
        Target frequency in Hz.
    z_in_real : float
        Re(Z_in) in Ω.
    z_in_imag : float
        Im(Z_in) in Ω.
    gamma_mag : float
        |Γ| (dimensionless).
    s11_db : float
        S11 in dB.
    v_eom_mag : float
        |V_EOM| RMS in V.
    i_source_rms : float
        |i_source_droop| RMS in A.
    power_balance_ok : bool
        Whether the power-balance check passed at this frequency.
    """

    frequency_hz: float = 0.0
    z_in_real: float = 0.0
    z_in_imag: float = 0.0
    gamma_mag: float = 0.0
    s11_db: float = 0.0
    v_eom_mag: float = 0.0
    i_source_rms: float = 0.0
    power_balance_ok: bool = False


class CoarseGridSummary(BaseModel):
    """Summary of coarse-grid evaluation.

    Attributes
    ----------
    coarse_evaluated : bool
        Whether the coarse grid was evaluated (lazy evaluation may skip it).
    off_target_n_points : int
        Number of off-target coarse grid points evaluated.
    off_target_v_eom_peak_v : float
        Peak |V_EOM| over off-target grid points (0 if not evaluated).
    """

    coarse_evaluated: bool = False
    off_target_n_points: int = 0
    off_target_v_eom_peak_v: float = 0.0


# ---------------------------------------------------------------------------
# Main result model
# ---------------------------------------------------------------------------


class CandidateResult(BaseModel):
    """Result data for a single optimizer candidate (spec §6.9).

    Mutable during construction; the tuple position in
    ``OptimizationResult.candidates`` conveys rank — no ``rank`` field
    is set on this model.

    Prompt-01 fields (topology_id, continuous_variables, etc.) are
    retained for backward compatibility.  Prompt-05 fields are additive
    with safe defaults.

    Attributes
    ----------
    candidate_id : str
        Unique identifier.
    topology_id : str
        Topology identifier (Prompt 01 field; domain_id preferred in Prompt 05).
    continuous_variables : dict[str, float]
        Optimized continuous variable values (Prompt 01 field; kept for
        backward compat; Prompt 05 stores k_residues / pole_frequencies instead).
    resolved_values : dict[str, float]
        Resolved L/C/R values in SI units.
    catalog_parts : dict[str, Any]
        Mapping from logical element to catalog part info (Prompt 09).
    objective_terms : dict[str, float]
        Individual objective term values (includes ``"total"``, ``"base"``,
        ``"soft_penalty"`` plus per-term breakdown).
    constraint_margins : dict[str, float]
        Normalized constraint margins (negative = violated).
    warnings : list[dict[str, Any]]
        Structured warnings from evaluation.
    solver_diagnostics : dict[str, Any]
        Solver-specific diagnostic data.
    feasible : bool
        Whether all hard constraints are satisfied and MNA succeeded.
    """

    # ---- Prompt 01 skeleton fields ----------------------------------------
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

    # ---- Prompt 05: domain identity ---------------------------------------
    orientation: str = ""
    domain_id: str = ""
    branch1_realization: str = ""
    branch2_realization: str = ""
    branch1_cells: int = 0
    branch2_cells: int = 0
    branch1_has_c0: bool = False
    branch1_has_linf: bool = False
    branch2_has_c0: bool = False
    branch2_has_linf: bool = False

    # ---- Prompt 05: Foster coefficients -----------------------------------
    k_residues_branch1: list[float] = Field(default_factory=list)
    k_residues_branch2: list[float] = Field(default_factory=list)
    k0_branch1: float | None = None
    k_inf_branch1: float | None = None
    k0_branch2: float | None = None
    k_inf_branch2: float | None = None
    pole_frequencies_branch1_hz: list[float] = Field(default_factory=list)
    pole_frequencies_branch2_hz: list[float] = Field(default_factory=list)

    # ---- Prompt 05: extended feasibility ----------------------------------
    near_feasible: bool = False
    v_max: float = 0.0
    v_sum: float = 0.0
    base_objective_value: float = 0.0
    soft_penalty_total: float = 0.0
    numerical_status: str = "ok"

    # ---- Prompt 05: circuit solution summaries (typed) --------------------
    target_solution_summaries: list[TargetSolutionSummary] = Field(default_factory=list)
    coarse_grid_summary: CoarseGridSummary = Field(default_factory=CoarseGridSummary)

    # ---- Prompt 05: provenance --------------------------------------------
    seed_source: str = ""
    de_domain_id: str = ""
    de_evaluations_used: int = 0
    de_generation_reached: int = 0
    pre_polish_objective: float | None = None
    local_polish_method: str = ""
    local_polish_success: bool = False
    local_polish_iterations: int = 0
    local_polish_evaluations: int = 0
    solver_termination: str = ""
