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


class DerivativeMode(enum.StrEnum):
    """Source of the derivatives supplied to the local solver (P12.5-E).

    ``REFERENCE_FD`` is the frozen P05 behaviour: SciPy differentiates the
    objective and the nonlinear constraints numerically.  ``ANALYTICAL`` supplies
    the validated P12.5-D ``DerivativeTransaction`` Jacobians instead.  The
    objective and constraint *values* are identical in both modes.
    """

    REFERENCE_FD = "reference_fd"
    ANALYTICAL = "analytical"


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

    # ---- Prompt 05 additions (backward-compatible; all have defaults) ----

    #: Maximum number of topology domains to send through DE.
    max_optimization_domains: int = Field(default=20, ge=1)
    #: SciPy DE mutation strategy string.
    de_strategy: str = "best1bin"
    #: Basin-deduplication radius in normalized [0,1]^n space (dim-normalized RMS).
    basin_dedup_radius: float = Field(default=0.05, gt=0.0, le=1.0)
    #: Hard-constraint feasibility tolerance (eps_feas).
    feasibility_tolerance: float = Field(default=1e-6, gt=0.0)
    #: Near-feasibility tolerance (eps_near), must be > feasibility_tolerance.
    near_feasibility_tolerance: float = Field(default=0.05, gt=0.0)
    #: Objective weight for reflection / match term J_gamma.
    objective_weight_gamma: float = Field(default=1.0, ge=0.0)
    #: Objective weight for EOM voltage tracking term J_voltage.
    objective_weight_voltage: float = Field(default=1.0, ge=0.0)
    #: Objective weight for parasitic loss term J_loss (0 = disabled by default).
    objective_weight_loss: float = Field(default=0.0, ge=0.0)
    #: Objective weight for topology complexity term J_complexity (constant within domain).
    objective_weight_complexity: float = Field(default=0.0, ge=0.0)

    # ---- P12.5-E addition (backward-compatible; default preserves frozen FD) ----

    #: Derivative source for local polish.  ``REFERENCE_FD`` is the frozen P05
    #: behaviour and remains the default; ``ANALYTICAL`` opts into the validated
    #: P12.5-D sensitivity transaction.
    local_derivative_mode: DerivativeMode = DerivativeMode.REFERENCE_FD


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

    # ---- Prompt 06: sweep band (explicit; None -> derived from targets) ----
    sweep_f_min_hz: float | None = None
    sweep_f_max_hz: float | None = None
    sweep_band_margin_lo: float = 0.5
    sweep_band_margin_hi: float = 2.0

    # ---- Prompt 06: adaptive sweep parameters ------------------------------
    sweep_n_base_points: int = 200
    sweep_max_depth: int = 5
    sweep_pole_min_refinement_depth: int = 3
    sweep_min_interval_width_ratio: float = 1e-3
    sweep_curvature_tol: float = 0.02
    sweep_off_target_v_eom_db_threshold: float = 6.0
    sweep_power_balance_rtol: float = 1e-3

    # ---- Prompt 06: Q extraction -------------------------------------------
    q_search_window_ratio: float = 0.3
    q_multiple_peak_threshold_ratio: float = 0.8
    q_usable_bw_eta: float = 0.9

    # ---- Prompt 06: time reconstruction ------------------------------------
    time_domain_min_samples_per_cycle: int = 20
    time_domain_max_n_points: int = 50_000
    time_domain_commensurate_rtol: float = 1e-4
    time_domain_commensurate_max_denominator: int = 1000
    time_domain_max_common_period_s: float = 1e-3
    time_domain_n_cycles_fallback: int = 10
    time_domain_mc_draws: int = 500
    time_domain_rng_seed: int | None = None


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
