"""Top-level optimization engine (Prompt 05).

Orchestrates: seeds → domain grouping → seed evaluation → domain ranking →
budget allocation → DE per domain → basin dedup → local polish → global ranking.

``run_optimization()`` does NOT write files.  Callers use ``save_results()``
from ``foster_eom.persistence.yaml_io`` for persistence.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.objectives import OptimizationSpec
from foster_eom.domain.results import CandidateResult, CoarseGridSummary, TargetSolutionSummary
from foster_eom.domain.source import SourceSpec
from foster_eom.foster.seed import SeedGenerationResult
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.de_runner import DEDiagnostics, run_de
from foster_eom.optimize.dedup import deb_key, deduplicate_basins
from foster_eom.optimize.domain import (
    ContinuousOptimizationDomain,
    group_seeds_into_domains,
)
from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    EvaluationContext,
    EvaluationResult,
    build_evaluation_context,
    evaluate,
)
from foster_eom.optimize.local_polish import polish_top_k
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.optimize.preflight import PreflightReport, run_preflight

# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """Complete run metadata for a single ``run_optimization()`` call."""

    foster_eom_version: str
    numpy_version: str
    scipy_version: str
    random_seed: int
    requested_global_budget: int
    seed_evaluation_budget_used: int
    de_budget_available: int
    allocated_budget_per_domain: dict[str, int]
    unique_x_evaluations_per_domain: dict[str, int]
    total_unique_x_evaluations: int
    budget_exhausted: bool
    n_domains_available: int
    n_domains_selected_before_budget: int
    n_domains_optimized: int
    n_domains_dropped_for_budget: int
    domain_search_truncated: bool


# ---------------------------------------------------------------------------
# Optimization result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationResult:
    """Complete result of ``run_optimization()``."""

    candidates: tuple[CandidateResult, ...]  # Deb-ranked (best first)
    best_feasible: CandidateResult | None
    near_feasible_best: CandidateResult | None
    preflight: PreflightReport
    seed_diagnostics: SeedGenerationResult
    de_diagnostics: tuple[DEDiagnostics, ...]
    run_manifest: RunManifest


# ---------------------------------------------------------------------------
# CandidateResult builder
# ---------------------------------------------------------------------------


def _build_candidate_result(
    result: EvaluationResult,
    domain: ContinuousOptimizationDomain,
    termination: str,
    seed_source: str = "foster_schmidt_04b",
    de_evaluations_used: int = 0,
    pre_polish_objective: float | None = None,
    polish_method: str = "",
    polish_success: bool = False,
    polish_iterations: int = 0,
    polish_evals: int = 0,
) -> CandidateResult:
    """Convert an EvaluationResult into a persisted CandidateResult."""

    # Per-target summaries
    summaries = []
    for sol in result.target_solutions:
        summaries.append(
            TargetSolutionSummary(
                frequency_hz=sol.f_hz,
                z_in_real=sol.z_in.real if sol.z_in else 0.0,
                z_in_imag=sol.z_in.imag if sol.z_in else 0.0,
                gamma_mag=abs(sol.gamma) if sol.gamma is not None else 0.0,
                s11_db=sol.s11_db or 0.0,
                v_eom_mag=abs(sol.v_eom) if sol.v_eom is not None else 0.0,
                i_source_rms=abs(sol.i_source_droop) if sol.i_source_droop is not None else 0.0,
                power_balance_ok=sol.power_balance_ok,
            )
        )

    # Coarse grid summary
    coarse_v_peak = 0.0
    coarse_n = 0
    if result.coarse_evaluated:
        for sol in result.all_solutions:
            if sol.v_eom is not None:
                coarse_v_peak = max(coarse_v_peak, abs(sol.v_eom))
        coarse_n = len(result.all_solutions) - len(result.target_solutions)

    coarse_summary = CoarseGridSummary(
        coarse_evaluated=result.coarse_evaluated,
        off_target_n_points=coarse_n,
        off_target_v_eom_peak_v=coarse_v_peak,
    )

    # Branch info
    topo = domain.topology

    # Objective terms
    obj_terms = dict(result.objective_terms)

    cr = CandidateResult(
        candidate_id=f"{domain.domain_id[:8]}_{id(result):x}",
        topology_id=domain.domain_id,
        # Foster branch info
        orientation=domain.orientation.value,
        domain_id=domain.domain_id,
        branch1_realization=domain.branch1_realization.value,
        branch2_realization=domain.branch2_realization.value,
        branch1_cells=topo.branch1_cells,
        branch2_cells=topo.branch2_cells,
        branch1_has_c0=topo.branch1_has_c0,
        branch1_has_linf=topo.branch1_has_linf,
        branch2_has_c0=topo.branch2_has_c0,
        branch2_has_linf=topo.branch2_has_linf,
        # Feasibility
        feasible=result.feasible,
        near_feasible=result.near_feasible,
        v_max=result.v_max,
        v_sum=result.v_sum,
        # Objectives
        objective_terms=obj_terms,
        base_objective_value=result.base_objective_value,
        soft_penalty_total=result.soft_penalty_total,
        # Constraints
        constraint_margins=dict(
            zip(
                [f"hard_{i}" for i in range(len(result.hard_margins))],
                result.hard_margins,
                strict=False,
            )
        ),
        # Circuit summaries
        target_solution_summaries=summaries,
        coarse_grid_summary=coarse_summary,
        # Numerical status
        numerical_status=result.numerical_status,
        # Provenance
        seed_source=seed_source,
        de_domain_id=domain.domain_id,
        de_evaluations_used=de_evaluations_used,
        pre_polish_objective=pre_polish_objective,
        local_polish_method=polish_method,
        local_polish_success=polish_success,
        local_polish_iterations=polish_iterations,
        local_polish_evaluations=polish_evals,
        solver_termination=termination,
    )
    return cr


# ---------------------------------------------------------------------------
# Domain budget allocation
# ---------------------------------------------------------------------------


def _allocate_budgets(
    domains: list[ContinuousOptimizationDomain],
    de_budget: int,
    pop_multiplier: int,
    domain_deb_keys: dict[str, tuple],
) -> tuple[dict[str, int], int, bool]:
    """Allocate DE budget per domain and drop if budget is exhausted.

    Returns (budget_per_domain, n_dropped, truncated).
    """

    # Minimum budget per domain
    def _min_budget(d: ContinuousOptimizationDomain) -> int:
        n_dim = d.dimension
        n_pop = max(4, pop_multiplier * max(n_dim, 1))
        return n_pop * 2

    # Check if total minimum fits
    total_min = sum(_min_budget(d) for d in domains)
    n_dropped = 0
    truncated = False

    working_domains = list(domains)
    while total_min > de_budget and len(working_domains) > 1:
        # Drop lowest-ranked domain (last in list, which is sorted best-first)
        working_domains.pop()
        n_dropped += 1
        total_min = sum(_min_budget(d) for d in working_domains)
        truncated = True

    if not working_domains:
        return {}, n_dropped, truncated

    # Allocate base budgets
    budget_map = {d.domain_id: _min_budget(d) for d in working_domains}
    remaining = de_budget - sum(budget_map.values())

    # Distribute remaining proportionally by rank score
    if remaining > 0:
        scores = [1.0 / (1.0 + i) for i in range(len(working_domains))]
        total_score = sum(scores)
        for i, d in enumerate(working_domains):
            extra = math.floor(remaining * scores[i] / total_score)
            budget_map[d.domain_id] += extra
        # Rounding residual to top-ranked domain
        allocated = sum(budget_map.values())
        budget_map[working_domains[0].domain_id] += de_budget - allocated

    return budget_map, n_dropped, truncated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_optimization(
    seed_result: SeedGenerationResult,
    opt_spec: OptimizationSpec,
    source_spec: SourceSpec,
    eom_model: OnePortModel,
    component_limits: ContinuousLimits,
    match_constraints: MatchConstraints,
    stress_constraints: StressConstraints,
    target_frequencies_hz: tuple[float, ...],
    sweep_f_min_hz: float,
    sweep_f_max_hz: float,
    base_grid_points: int = 200,
    voltage_targets_rms_v: tuple[float | None, ...] = (),
    extra_constraint_records: list | None = None,
    checkpoint_path: str | Path | None = None,
    warm_start_candidates: list[CandidateResult] | None = None,
) -> OptimizationResult:
    """Run the full Prompt-05 layered optimization pipeline.

    Does NOT write any files.  Call ``save_results()`` from
    ``foster_eom.persistence.yaml_io`` to persist results.

    Parameters
    ----------
    seed_result : SeedGenerationResult
        Output from the 04B seed generator.
    opt_spec : OptimizationSpec
        Optimization configuration.
    source_spec : SourceSpec
        Thévenin source specification.
    eom_model : OnePortModel
        EOM load model.
    component_limits : ContinuousLimits
        Physical component limits.
    match_constraints : MatchConstraints
        Impedance match constraints.
    stress_constraints : StressConstraints
        Component stress limits.
    target_frequencies_hz : tuple[float, ...]
        Target frequencies.
    sweep_f_min_hz, sweep_f_max_hz : float
        Coarse sweep band.
    base_grid_points : int
        Number of coarse grid points.
    voltage_targets_rms_v : tuple[float | None, ...]
        Per-target EOM voltage targets (None = no target).
    extra_constraint_records : list | None
        Additional ``ConstraintRecord`` entries.
    """
    import importlib.metadata

    # ---- 0. Preflight ----
    preflight = run_preflight(opt_spec)

    def _version(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    # ---- 1. Domain grouping ----
    from foster_eom.domain.topology import TopologySearchSpec

    topo_spec = TopologySearchSpec()  # default spec — caller should provide if needed
    from foster_eom.foster.seed import _domain_to_internal_pole_spec

    # Use the pole specs from the seed result's topology spec if available
    # (best effort — production callers should pass the topo_spec directly)
    pole_spec_b1 = _domain_to_internal_pole_spec(topo_spec.pole_spec_branch1)
    pole_spec_b2 = _domain_to_internal_pole_spec(topo_spec.pole_spec_branch2)

    all_domains = group_seeds_into_domains(
        seeds=seed_result.seeds,
        pole_spec_b1=pole_spec_b1,
        pole_spec_b2=pole_spec_b2,
        f_targets_hz=np.array(target_frequencies_hz),
        component_limits=component_limits,
    )

    feasible_domains = [d for d in all_domains if d.structurally_feasible]

    # ---- 2. Objective config ----
    obj_config = ObjectiveConfig(
        z_ref_ohm=source_spec.z_ref_ohm,
        w_gamma=opt_spec.objective_weight_gamma,
        w_voltage=opt_spec.objective_weight_voltage,
        w_loss=opt_spec.objective_weight_loss,
        w_complexity=opt_spec.objective_weight_complexity,
        voltage_targets_rms_v=voltage_targets_rms_v,
        voltage_target_weights=tuple(1.0 for _ in voltage_targets_rms_v),
    )

    # ---- 3. Evaluate all analytic seeds per domain ----
    domain_seed_results: dict[str, list[EvaluationResult]] = {}
    all_caches: dict[str, DomainEvaluatorCache] = {}
    all_contexts: dict[str, EvaluationContext] = {}
    total_seed_evals = 0

    for d in feasible_domains:
        cache = DomainEvaluatorCache()
        ctx = build_evaluation_context(
            domain=d,
            source_spec=source_spec,
            eom_model=eom_model,
            component_limits=component_limits,
            match_constraints=match_constraints,
            stress_constraints=stress_constraints,
            target_frequencies_hz=target_frequencies_hz,
            sweep_f_min_hz=sweep_f_min_hz,
            sweep_f_max_hz=sweep_f_max_hz,
            base_grid_points=base_grid_points,
            objective_config=obj_config,
            feasibility_tolerance=opt_spec.feasibility_tolerance,
            near_feasibility_tolerance=opt_spec.near_feasibility_tolerance,
            extra_constraint_records=extra_constraint_records,
        )
        all_caches[d.domain_id] = cache
        all_contexts[d.domain_id] = ctx

        seed_results: list[EvaluationResult] = []
        for si in d.seed_indices:
            seed = seed_result.seeds[si]
            # Pack seed into x
            b1_solve = seed.branch1_solve
            b2_solve = seed.branch2_solve
            x_vec = d.variable_mapper.pack(
                k0_b1=b1_solve.k0 if b1_solve else None,
                k_inf_b1=b1_solve.k_inf if b1_solve else None,
                k_residues_b1=b1_solve.k_residues if b1_solve else (),
                f_poles_b1=b1_solve.f_poles_hz if b1_solve else (),
                k0_b2=b2_solve.k0 if b2_solve else None,
                k_inf_b2=b2_solve.k_inf if b2_solve else None,
                k_residues_b2=b2_solve.k_residues if b2_solve else (),
                f_poles_b2=b2_solve.f_poles_hz if b2_solve else (),
            )
            r = evaluate(x_vec, ctx, cache)
            seed_results.append(r)

        domain_seed_results[d.domain_id] = seed_results
        total_seed_evals += cache.n_unique_evaluations

    # ---- 4. Domain selection by Prompt-05 Deb key ----
    def _domain_deb_key(d: ContinuousOptimizationDomain) -> tuple:
        results = domain_seed_results.get(d.domain_id, [])
        if not results:
            return (True, 1.0, 1.0, 1e9, d.domain_id)
        best = min(results, key=deb_key)
        return (*deb_key(best), d.domain_id)

    # Topology-family rank
    seen_families: dict[tuple, int] = {}
    family_ranks: dict[str, int] = {}
    for d in feasible_domains:
        topo = d.topology
        fam = (
            topo.branch1_cells,
            topo.branch2_cells,
            topo.branch1_has_c0,
            topo.branch1_has_linf,
            topo.branch2_has_c0,
            topo.branch2_has_linf,
        )
        if fam not in seen_families:
            seen_families[fam] = len(seen_families)
        family_ranks[d.domain_id] = seen_families[fam]

    def _selection_key(d: ContinuousOptimizationDomain) -> tuple:
        ori_idx = 0  # simplistic orientation sort index
        return (ori_idx, family_ranks.get(d.domain_id, 99), _domain_deb_key(d))

    sorted_domains = sorted(feasible_domains, key=_selection_key)
    n_available = len(sorted_domains)

    # Apply max_optimization_domains cap
    selected_domains = sorted_domains[: opt_spec.max_optimization_domains]
    n_selected_before_budget = len(selected_domains)

    # ---- 5. Budget allocation ----
    de_budget_available = max(0, opt_spec.max_global_evaluations - total_seed_evals)
    domain_deb_keys = {d.domain_id: _domain_deb_key(d) for d in selected_domains}
    budget_map, n_dropped, truncated = _allocate_budgets(
        selected_domains, de_budget_available, opt_spec.population_size_multiplier, domain_deb_keys
    )
    optimized_domains = [d for d in selected_domains if d.domain_id in budget_map]

    # ---- 6. DE per domain ----
    all_de_diags: list[DEDiagnostics] = []
    all_candidate_results: list[CandidateResult] = []
    total_unique_evals = total_seed_evals

    for domain in optimized_domains:
        cache = all_caches[domain.domain_id]
        ctx = all_contexts[domain.domain_id]
        seed_res = domain_seed_results[domain.domain_id]
        budget = budget_map[domain.domain_id]

        # Baseline: analytic seed
        analytic_best = min(seed_res, key=deb_key) if seed_res else None

        def _build_checkpoint(cache: DomainEvaluatorCache = cache, domain: ContinuousOptimizationDomain = domain) -> None:
            if not checkpoint_path:
                return

            from foster_eom.persistence.yaml_io import save_results

            # Temporary collect candidates
            current_cands = []
            for r in all_candidate_results:
                current_cands.append(r)

            # Plus best of current domain so far
            best_curr = min(cache._cache.values(), key=deb_key) if cache._cache else None
            if best_curr:
                ccr = _build_candidate_result(
                    result=best_curr,
                    domain=domain,
                    termination="checkpoint",
                    de_evaluations_used=cache.n_unique_evaluations,
                )
                current_cands.append(ccr)

            current_cands.sort(
                key=lambda c: (
                    not c.feasible,
                    c.v_max,
                    c.v_sum,
                    c.objective_terms.get("total", 1e9),
                )
            )

            manifest_temp = RunManifest(
                foster_eom_version="unknown",
                numpy_version="unknown",
                scipy_version="unknown",
                random_seed=opt_spec.random_seed,
                requested_global_budget=opt_spec.max_global_evaluations,
                seed_evaluation_budget_used=total_seed_evals,
                de_budget_available=de_budget_available,
                allocated_budget_per_domain=budget_map,
                unique_x_evaluations_per_domain={},
                total_unique_x_evaluations=0,
                budget_exhausted=False,
                n_domains_available=n_available,
                n_domains_selected_before_budget=n_selected_before_budget,
                n_domains_optimized=len(optimized_domains),
                n_domains_dropped_for_budget=n_dropped,
                domain_search_truncated=truncated,
            )
            res_temp = OptimizationResult(
                candidates=tuple(current_cands),
                best_feasible=None,
                near_feasible_best=None,
                preflight=preflight,
                seed_diagnostics=seed_result,
                de_diagnostics=tuple(all_de_diags),
                run_manifest=manifest_temp,
            )
            with contextlib.suppress(Exception):
                save_results(res_temp, checkpoint_path)

        de_candidates, de_diag = run_de(
            context=ctx,
            cache=cache,
            analytic_seed_results=seed_res,
            budget=budget,
            population_size_multiplier=opt_spec.population_size_multiplier,
            random_seed=opt_spec.random_seed,
            de_strategy=opt_spec.de_strategy,
            workers=opt_spec.workers,
            warm_start_candidates=warm_start_candidates,
            checkpoint_interval=opt_spec.checkpoint_every_evaluations,
            checkpoint_callback=_build_checkpoint,
        )
        all_de_diags.append(de_diag)

        # ---- 7. Basin dedup ----
        basins = deduplicate_basins(de_candidates, radius=opt_spec.basin_dedup_radius)

        # ---- 8. Local polish ----
        polish_results = polish_top_k(basins, ctx, cache, opt_spec)

        # Collect polished candidates
        polished_set = list({pr.retained.x: pr.retained for pr in polish_results}.values())

        # All unique candidates (polished retained + rest of basins)
        domain_final: list[EvaluationResult] = []
        for pr in polish_results:
            domain_final.append(pr.retained)
        for b in basins:
            if b.representative not in polished_set:
                domain_final.append(b.representative)

        # ---- 9. Baseline protection ----
        if analytic_best is not None:
            domain_final.append(analytic_best)

        # Sort by Deb key
        domain_final.sort(key=deb_key)

        # Build CandidateResult for each
        polish_map = {pr.pre_polish.x: pr for pr in polish_results}

        for res in domain_final:
            polish_res = polish_map.get(res.x)
            cr = _build_candidate_result(
                result=res,
                domain=domain,
                termination=de_diag.de_termination,
                de_evaluations_used=cache.n_unique_evaluations,
                pre_polish_objective=polish_res.pre_polish.objective_value if polish_res else None,
                polish_method=polish_res.method_used if polish_res else "",
                polish_success=polish_res.success if polish_res else False,
                polish_iterations=polish_res.n_iterations if polish_res else 0,
                polish_evals=polish_res.n_evaluations if polish_res else 0,
            )
            all_candidate_results.append(cr)

        total_unique_evals += cache.n_unique_evaluations - total_seed_evals

    # ---- 10. Global ranking ----
    def _cr_deb_key(cr: CandidateResult) -> tuple:
        return (
            not cr.feasible,
            cr.v_max,
            cr.v_sum,
            cr.objective_terms.get("total", 1e9),
        )

    all_candidate_results.sort(key=_cr_deb_key)

    best_feasible = next((c for c in all_candidate_results if c.feasible), None)
    near_feas_best = next((c for c in all_candidate_results if c.near_feasible), None)

    import numpy as np_v
    import scipy as sp_v

    try:
        npm = np_v.__version__
    except Exception:
        npm = "unknown"
    try:
        spm = sp_v.__version__
    except Exception:
        spm = "unknown"

    manifest = RunManifest(
        foster_eom_version=_version("foster_eom"),
        numpy_version=npm,
        scipy_version=spm,
        random_seed=opt_spec.random_seed,
        requested_global_budget=opt_spec.max_global_evaluations,
        seed_evaluation_budget_used=total_seed_evals,
        de_budget_available=de_budget_available,
        allocated_budget_per_domain=budget_map,
        unique_x_evaluations_per_domain={
            d.domain_id: all_caches[d.domain_id].n_unique_evaluations for d in optimized_domains
        },
        total_unique_x_evaluations=total_unique_evals,
        budget_exhausted=total_unique_evals >= opt_spec.max_global_evaluations,
        n_domains_available=n_available,
        n_domains_selected_before_budget=n_selected_before_budget,
        n_domains_optimized=len(optimized_domains),
        n_domains_dropped_for_budget=n_dropped,
        domain_search_truncated=truncated,
    )

    return OptimizationResult(
        candidates=tuple(all_candidate_results),
        best_feasible=best_feasible,
        near_feasible_best=near_feas_best,
        preflight=preflight,
        seed_diagnostics=seed_result,
        de_diagnostics=tuple(all_de_diags),
        run_manifest=manifest,
    )
