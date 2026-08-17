"""Top-level orchestrator for discrete catalog realization (Prompt 09).

``realize()`` — main entry point.

Workflow:
    1. Extract per-slot continuous values from BranchCoordinates.
    2. Build neighborhoods (per-slot catalog queries).
    3. Generate combinations (exhaustive or beam).
    4. For each combo: build catalog OnePortModels, substitute into graph,
       evaluate with P05 MNA infrastructure.
    5. Rank combos by exact P05 Deb key.
    6. Run P06 verification on top-K in Deb order.
    7. Return RealizationResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from foster_eom.optimize.dedup import deb_key
from foster_eom.realization.beam import Combo, generate_combos
from foster_eom.realization.neighborhoods import build_neighborhoods, build_slot_specs
from foster_eom.realization.result import (
    CatalogCombo,
    RealizationDiagnostics,
    RealizationResult,
)
from foster_eom.realization.spec import (
    NeighborhoodEntry,
    RealizationBudget,
    RealizationSpec,
)
from foster_eom.realization.substitute import evaluate_with_overrides

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.circuit.graph import CircuitGraph
    from foster_eom.optimize.evaluator import EvaluationContext, EvaluationResult
    from foster_eom.optimize.variable_map import BranchCoordinates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def realize(
    continuous_result: EvaluationResult,
    context: EvaluationContext,
    b1: BranchCoordinates,
    b2: BranchCoordinates,
    base_graph: CircuitGraph,
    library: ComponentLibrary,
    spec: RealizationSpec | None = None,
    budget: RealizationBudget | None = None,
) -> RealizationResult:
    """Convert a continuous candidate into discrete catalog realizations.

    Parameters
    ----------
    continuous_result : EvaluationResult
        The frozen P05 continuous evaluation (baseline).
    context : EvaluationContext
        Frozen evaluation context (frequencies, constraints, objective).
    b1, b2 : BranchCoordinates
        Unpacked branch coordinates from the continuous result.
    base_graph : CircuitGraph
        The continuous circuit graph (primitive L/C elements).
    library : ComponentLibrary
        Opened P08 component library.
    spec : RealizationSpec | None
        Realization configuration.  If None, auto-built with defaults.
    budget : RealizationBudget | None
        MNA solve budget.  If None, uses default (512 solves).

    Returns
    -------
    RealizationResult
    """
    if budget is None:
        budget = RealizationBudget()

    # ------------------------------------------------------------------
    # 1. Build slot specs (auto if not provided)
    # ------------------------------------------------------------------
    if spec is None:
        auto_slots = build_slot_specs(context, b1, b2)
        spec = RealizationSpec(slot_specs=auto_slots)
    elif not spec.slot_specs:
        auto_slots = build_slot_specs(context, b1, b2)
        spec = RealizationSpec(
            slot_specs=auto_slots,
            k_max=spec.k_max,
            exhaustive_threshold=spec.exhaustive_threshold,
            beam_width=spec.beam_width,
            random_seed=spec.random_seed,
            combination_mode=spec.combination_mode,
            verify_top_k=spec.verify_top_k,
        )

    # ------------------------------------------------------------------
    # 2. Build neighborhoods
    # ------------------------------------------------------------------
    neighborhoods = build_neighborhoods(spec.slot_specs, library, k_max=spec.k_max)

    # Check for empty slots
    failed_slots = [eid for eid, entries in neighborhoods.items() if not entries]
    if failed_slots:
        diag = RealizationDiagnostics(
            n_slots=len(spec.slot_specs),
            parts_per_slot={eid: len(e) for eid, e in neighborhoods.items()},
            total_combos=0,
            n_combos_generated=0,
            n_combos_evaluated=0,
            n_mna_solves=0,
            search_exhaustive=False,
            search_truncated=False,
            budget_exhausted=False,
        )
        return RealizationResult(
            status="no_candidates",
            continuous_baseline=continuous_result,
            failed_slots=failed_slots,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------
    # 3. Generate combinations
    # ------------------------------------------------------------------
    combos, search_exhaustive, search_truncated = generate_combos(neighborhoods, spec)

    parts_per_slot = {eid: len(e) for eid, e in neighborhoods.items()}
    total_combos = 1
    for c in parts_per_slot.values():
        total_combos *= c

    # ------------------------------------------------------------------
    # 4. Evaluate combinations
    # ------------------------------------------------------------------
    catalog_combos: list[CatalogCombo] = []
    n_evaluated = 0

    for combo in combos:
        if budget.exhausted:
            break

        cat_combo = _evaluate_combo(combo, base_graph, context, library, spec, budget)
        if cat_combo is not None:
            catalog_combos.append(cat_combo)
            n_evaluated += 1

    # ------------------------------------------------------------------
    # 5. Rank by P05 Deb key
    # ------------------------------------------------------------------
    catalog_combos.sort(key=lambda cc: cc.deb_key)

    # ------------------------------------------------------------------
    # 6. P06 verification on top-K
    # ------------------------------------------------------------------
    verified: list[CatalogCombo] = []
    first_passing: CatalogCombo | None = None

    for cc in catalog_combos[: spec.verify_top_k]:
        _run_p06_verify(cc, base_graph, context, library)
        verified.append(cc)
        if cc.verify_passed and first_passing is None:
            first_passing = cc

    # ------------------------------------------------------------------
    # 7. Determine status and best combo
    # ------------------------------------------------------------------
    # best = first P06-verified passing combo (when one exists), otherwise
    # the Deb-best of all evaluated combos regardless of verification.
    # This ensures that if Deb-#1 fails P06 and Deb-#2 passes, best = Deb-#2.
    if first_passing is not None:
        best: CatalogCombo | None = first_passing
    elif catalog_combos:
        best = catalog_combos[0]  # Deb-best unverified fallback
    else:
        best = None

    n_unverified = len(catalog_combos) - len(verified)
    all_verified_failed = verified and all(not cc.verify_passed for cc in verified)

    if not catalog_combos:
        status = "no_candidates"
    elif first_passing is not None:
        status = "feasible"
    elif search_exhaustive and all(not cc.eval_result.feasible for cc in catalog_combos):
        # All MNA-infeasible AND exhaustive → genuinely infeasible
        status = "infeasible"
    elif all_verified_failed and n_unverified == 0 and search_exhaustive:
        # All combos verified, all failed, and search was exhaustive
        status = "infeasible"
    elif best is not None and best.eval_result.near_feasible:
        status = "degraded"
    else:
        # Covers: truncated search, partial verification, unverified candidates remaining
        status = "no_feasible_found"

    degradation: float | None = None
    if best is not None:
        degradation = best.eval_result.objective_value - continuous_result.objective_value

    diag = RealizationDiagnostics(
        n_slots=len(spec.slot_specs),
        parts_per_slot=parts_per_slot,
        total_combos=total_combos,
        n_combos_generated=len(combos),
        n_combos_evaluated=n_evaluated,
        n_mna_solves=budget.used,
        search_exhaustive=search_exhaustive,
        search_truncated=search_truncated,
        budget_exhausted=budget.exhausted,
    )

    return RealizationResult(
        status=status,
        continuous_baseline=continuous_result,
        combos=catalog_combos,
        best=best,
        degradation=degradation,
        failed_slots=[],
        diagnostics=diag,
        verified_combos=verified,
        first_passing_combo=first_passing,
    )


# ---------------------------------------------------------------------------
# Combo evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_combo(
    combo: Combo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
    spec: RealizationSpec,
    budget: RealizationBudget,
) -> CatalogCombo | None:
    """Build OnePortModels for all slots and evaluate the substituted graph."""

    slot_entries: dict[str, NeighborhoodEntry] = {}
    model_overrides: dict[str, Any] = {}

    for element_id, entry in combo:
        slot_entries[element_id] = entry

        # Build model from the frozen model_condition_id
        try:
            slot = next(s for s in spec.slot_specs if s.element_id == element_id)
            model = library.build_model(
                entry.component_id,
                freq_range=slot.freq_range_hz,
                fallback=slot.fallback_policy,
            )
        except Exception:
            # Model construction failure → skip combo
            return None

        model_overrides[element_id] = model

    # Evaluate
    try:
        eval_result = evaluate_with_overrides(
            base_graph,
            model_overrides,
            context,
            budget=budget,
        )
    except Exception:
        return None

    dk = deb_key(eval_result)
    return CatalogCombo(
        slot_entries=slot_entries,
        eval_result=eval_result,
        deb_key=dk,
    )


# ---------------------------------------------------------------------------
# P06 verification helper
# ---------------------------------------------------------------------------


def _run_p06_verify(
    cc: CatalogCombo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
) -> None:
    """Run P06 adaptive sweep on the substituted graph and annotate cc."""
    from foster_eom.analysis.sweep import SweepSpec, compute_adaptive_sweep
    from foster_eom.catalog.component import FallbackPolicy

    # Rebuild model overrides for this combo
    model_overrides: dict[str, Any] = {}
    for element_id, entry in cc.slot_entries.items():
        try:
            model = library.build_model(
                entry.component_id,
                fallback=FallbackPolicy.ALLOW_LOWER_TIER,
            )
            model_overrides[element_id] = model
        except Exception:
            cc.verify_passed = False
            cc.verify_report = {"error": f"model build failed for {element_id}"}
            return

    try:
        from foster_eom.realization.substitute import build_substituted_graph

        subst_graph = build_substituted_graph(base_graph, model_overrides)  # type: ignore[arg-type]
    except Exception as exc:
        cc.verify_passed = False
        cc.verify_report = {"error": f"graph substitution failed: {exc}"}
        return

    # Derive sweep band.
    #
    # Use the explicitly resolved P06 band from ctx when available.  Otherwise
    # derive from targets with from_targets() default margins — critically, do NOT
    # pass validity_range=eval_frequencies_hz because that would clip the P06
    # continuous sweep to the discrete P05 grid, which may be narrower than the
    # margin-expanded band that P06 needs.
    #
    # Model eligibility (build_slot_specs) is separately constrained to
    # (min, max)(eval_frequencies_hz), which is the correct P05 band.
    target_hz = tuple(context.evaluation_frequencies_hz[i] for i in context.target_indices)

    if context.p06_sweep_band_hz is not None:
        # Explicit resolved band — use it directly
        f_min, f_max = context.p06_sweep_band_hz
        sweep_spec = SweepSpec(f_min_hz=f_min, f_max_hz=f_max)
    else:
        # Derive from targets using margin factors, no validity_range clipping
        sweep_spec = SweepSpec.from_targets(target_hz=target_hz)

    try:
        sweep_result = compute_adaptive_sweep(
            graph=subst_graph,
            source_spec=context.source_spec,
            eom_model=context.eom_model,
            spec=sweep_spec,
            target_hz=target_hz,
        )
        failed = sweep_result.failed_frequencies_hz
        passed = sweep_result.verification_complete and not failed
        cc.verify_passed = passed
        cc.verify_report = {
            "verification_complete": sweep_result.verification_complete,
            "n_evaluations": len(sweep_result.frequencies_hz),
            "failed_frequencies_hz": list(failed),
            "resonance_peaks": len(sweep_result.resonance_list),
            "off_target_unsafe": sweep_result.off_target_unsafe,
        }
    except Exception as exc:
        cc.verify_passed = False
        cc.verify_report = {"error": str(exc)}
