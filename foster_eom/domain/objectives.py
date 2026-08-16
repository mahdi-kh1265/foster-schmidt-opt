"""Optimization, analysis, and export specifications (spec §6.8)."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class GlobalMethod(enum.StrEnum):
    """Global optimization method."""

    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class LocalMethod(enum.StrEnum):
    """Local optimization method."""

    IPOPT = "ipopt"
    TRUST_CONSTR = "trust-constr"
    SLSQP = "slsqp"


class OptimizationPreset(enum.StrEnum):
    """Pre-configured optimization intensity levels."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    CUSTOM = "custom"


class OptimizationSpec(BaseModel, frozen=True):
    """Optimization configuration (spec §6.8).

    Attributes
    ----------
    preset : OptimizationPreset
        Convenience preset.
    random_seed : int
        RNG seed for reproducibility.
    global_method : GlobalMethod
        Global search method.
    population_size_multiplier : int
        DE population = multiplier x number of variables.
    max_global_evaluations : int
        Maximum objective evaluations in global search.
    workers : int | str
        Number of parallel workers (int or ``"auto"``).
    polish_top_k : int
        Number of best global candidates to polish locally.
    local_method : LocalMethod
        Preferred local solver.
    local_fallback_method : LocalMethod
        Fallback if preferred is unavailable.
    local_max_iterations : int
        Maximum local solver iterations.
    finite_difference_step : float
        Step for numerical gradients.
    feasibility_first : bool
        Prioritize feasibility over objective.
    stagnation_limit : int | None
        Stop after this many evals without improvement.
    checkpoint_every_evaluations : int
        Checkpoint interval.
    """

    preset: OptimizationPreset = OptimizationPreset.BALANCED
    random_seed: int = 20260815
    global_method: GlobalMethod = GlobalMethod.DIFFERENTIAL_EVOLUTION
    population_size_multiplier: int = Field(default=12, ge=1)
    max_global_evaluations: int = Field(default=50_000, ge=100)
    workers: int | str = "auto"
    polish_top_k: int = Field(default=8, ge=1)
    local_method: LocalMethod = LocalMethod.IPOPT
    local_fallback_method: LocalMethod = LocalMethod.TRUST_CONSTR
    local_max_iterations: int = Field(default=1500, ge=1)
    finite_difference_step: float = Field(default=1.0e-7, gt=0.0)
    feasibility_first: bool = True
    stagnation_limit: int | None = Field(default=None, ge=1)
    checkpoint_every_evaluations: int = Field(default=5000, ge=100)


class TimeDomainPhaseMode(enum.StrEnum):
    """Phase assumption for time-domain reconstruction."""

    SPECIFIED = "specified"
    ALL_ZERO = "all_zero"
    RANDOM_MC = "random_mc"
    WORST_CASE = "worst_case"
    CONSERVATIVE_BOUND = "conservative_bound"


class SpiceVerificationMode(enum.StrEnum):
    """When to run SPICE verification."""

    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


class AnalysisSpec(BaseModel, frozen=True):
    """Post-optimization analysis configuration.

    Attributes
    ----------
    detect_unintended_resonances : bool
        Search for off-target peaks.
    time_domain_reconstruction : bool
        Perform multi-tone time reconstruction.
    time_domain_phase_mode : TimeDomainPhaseMode
        Phase assumption for time-domain analysis.
    spice_verification : SpiceVerificationMode
        SPICE cross-check behavior.
    """

    detect_unintended_resonances: bool = True
    time_domain_reconstruction: bool = True
    time_domain_phase_mode: TimeDomainPhaseMode = TimeDomainPhaseMode.SPECIFIED
    spice_verification: SpiceVerificationMode = SpiceVerificationMode.OPTIONAL


class ExportSpec(BaseModel, frozen=True):
    """Export configuration.

    Attributes
    ----------
    save_csv : bool
        Export CSV frequency response.
    save_npz : bool
        Save NumPy arrays.
    save_spice_netlist : bool
        Generate SPICE netlist.
    save_plots : bool
        Generate diagnostic plots.
    save_full_provenance : bool
        Include complete provenance in bundle.
    """

    save_csv: bool = True
    save_npz: bool = True
    save_spice_netlist: bool = True
    save_plots: bool = True
    save_full_provenance: bool = True
