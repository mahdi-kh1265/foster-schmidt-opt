"""P10 top-level orchestrator: run_robustness().

Workflow:
  1. build_slot_uncertainties() from frozen P09 CatalogCombo.
  2. draw_samples() → N x D uniform matrix.
  3. inverse_transform_draw() per sample → physical values.
  4. evaluate_sample() for each sample → SampleResult with 4-state outcome.
  5. run_p06_diagnostic() on targeted samples.
  6. compute_yield_stats() + compute_distributions().
  7. run_oat_sensitivity() + compute_failure_association().
  8. Assemble RobustnessResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.circuit.graph import CircuitGraph
    from foster_eom.optimize.evaluator import EvaluationContext
    from foster_eom.realization.result import CatalogCombo
    from foster_eom.robustness.result import RobustnessResult
    from foster_eom.robustness.sampler import RobustnessSpec


def run_robustness(
    combo: CatalogCombo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
    spec: RobustnessSpec | None = None,
    op_condition_overrides: dict[str, float] | None = None,
    model_uncertainty_overrides: dict[str, float] | None = None,
) -> RobustnessResult:
    """Full P10 robustness analysis for a frozen P09 CatalogCombo.

    Parameters
    ----------
    combo : CatalogCombo
        Frozen P09 best realization (selected parts + models).
    base_graph : CircuitGraph
        Continuous circuit graph (primitive L/C elements).
    context : EvaluationContext
        Frozen evaluation context.
    library : ComponentLibrary
        Opened P08 component library.
    spec : RobustnessSpec | None
        Analysis configuration. Defaults to 500 iid-random samples.
    op_condition_overrides : dict[str, float] | None
        User-supplied operating-condition fractional tolerance per element_id.
        Never auto-generated.
    model_uncertainty_overrides : dict[str, float] | None
        User-supplied model-uncertainty fractional tolerance per element_id.
        Never auto-generated.

    Returns
    -------
    RobustnessResult
    """
    from foster_eom.robustness.evaluator import evaluate_sample
    from foster_eom.robustness.p06_strategy import run_p06_diagnostic
    from foster_eom.robustness.result import RobustnessResult
    from foster_eom.robustness.sampler import RobustnessSpec as _RSpec
    from foster_eom.robustness.sampler import (
        draw_samples,
        inverse_transform_draw,
    )
    from foster_eom.robustness.sensitivity import compute_failure_association, run_oat_sensitivity
    from foster_eom.robustness.stats import compute_distributions, compute_yield_stats
    from foster_eom.robustness.uncertainty import PerturbMethod, build_slot_uncertainties

    if spec is None:
        spec = _RSpec()

    # 1. Build slot uncertainties
    slot_uncertainties = build_slot_uncertainties(
        combo,
        op_condition_overrides=op_condition_overrides,
        model_uncertainty_overrides=model_uncertainty_overrides,
    )
    non_stochastic = [su.element_id for su in slot_uncertainties if not su.is_stochastic]

    # Collect perturbation notes for measured_residual slots
    perturbation_notes: list[str] = []
    for su in slot_uncertainties:
        if su.is_stochastic and su.perturb_method == PerturbMethod.MEASURED_RESIDUAL:
            eid_upper = su.element_id.upper()
            kind = "inductor" if ("_L" in eid_upper or "LINF" in eid_upper) else "capacitor"
            perturbation_notes.append(
                f"{su.element_id}: measured_residual first-order correction applied "
                f"(approximation; {kind} reactive term added to measured Z). "
                "Manufacturing tolerance source preserved."
            )

    # Nominal values from combo entries
    nominal_values = {eid: entry.value_nom for eid, entry in combo.slot_entries.items()}

    # 2. Draw sample matrix
    draw_matrix = draw_samples(slot_uncertainties, spec)

    # 3 + 4. Evaluate each sample
    samples = []
    for i in range(spec.n_samples):
        import numpy as np
        u_row = draw_matrix.u[i] if draw_matrix.u.shape[1] > 0 else np.array([])

        draw = inverse_transform_draw(u_row, slot_uncertainties, nominal_values)
        sample = evaluate_sample(
            sample_idx=i,
            draw=draw,
            slot_uncertainties=slot_uncertainties,
            combo=combo,
            base_graph=base_graph,
            context=context,
            library=library,
            spec=spec,
        )
        samples.append(sample)

    # 5. P06 diagnostic
    p06_all_run = spec.p06_diagnostic == "all"
    p06_results, p06_label = run_p06_diagnostic(
        samples=samples,
        combo=combo,
        base_graph=base_graph,
        context=context,
        library=library,
        spec=spec,
        slot_uncertainties=slot_uncertainties,
    )

    # 6. Statistics
    yield_stats = compute_yield_stats(samples, spec, p06_all_run=p06_all_run)
    distributions = compute_distributions(samples)

    # 7. Sensitivity
    oat = run_oat_sensitivity(
        combo=combo,
        base_graph=base_graph,
        context=context,
        library=library,
        slot_uncertainties=slot_uncertainties,
    )
    assoc = compute_failure_association(samples, slot_uncertainties)

    # Nominal baseline
    nominal_obj = combo.eval_result.objective_value
    nominal_feas = combo.eval_result.feasible
    nominal_verify = combo.verify_passed

    return RobustnessResult(
        spec=spec,
        combo=combo,
        slot_uncertainties=slot_uncertainties,
        non_stochastic_slots=non_stochastic,
        perturbation_notes=perturbation_notes,
        samples=samples,
        yield_stats=yield_stats,
        distributions=distributions,
        oat_sensitivity=oat,
        failure_association=assoc,
        p06_diagnostic_results=p06_results,
        p06_diagnostic_label=p06_label,
        nominal_objective=nominal_obj,
        nominal_feasible=nominal_feas,
        nominal_verify_passed=nominal_verify,
    )
