"""P10 P06 diagnostic strategies.

Runs P06 adaptive sweep on a targeted subset of samples.
This is a diagnostic tool — NOT a population yield estimator.
yield_p06 is only set when p06_diagnostic="all".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.circuit.graph import CircuitGraph
    from foster_eom.optimize.evaluator import EvaluationContext
    from foster_eom.realization.result import CatalogCombo
    from foster_eom.robustness.evaluator import SampleResult
    from foster_eom.robustness.sampler import RobustnessSpec


def run_p06_diagnostic(
    samples: list[SampleResult],
    combo: CatalogCombo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
    spec: RobustnessSpec,
    slot_uncertainties: list,
) -> tuple[list[SampleResult], str]:
    """Run P06 adaptive sweep on targeted samples per spec.p06_diagnostic.

    Returns
    -------
    (verified_samples, label)
        verified_samples: subset of samples with verify_report / verify_passed set.
        label: human-readable description of which samples were verified.
    """
    from foster_eom.robustness.evaluator import SampleOutcome

    if spec.p06_diagnostic == "none":
        return [], "none"

    if spec.p06_diagnostic == "worst_k":
        # Select k worst-objective PHYSICAL_FAIL + PASS samples (highest objective)
        candidates = [
            s
            for s in samples
            if s.outcome in (SampleOutcome.PASS, SampleOutcome.PHYSICAL_FAIL)
            and s.objective_value is not None
        ]
        candidates_sorted = sorted(candidates, key=lambda s: s.objective_value or 0.0, reverse=True)
        target = candidates_sorted[: spec.p06_worst_k]
        label = f"worst_{len(target)}_by_objective"

    elif spec.p06_diagnostic == "all":
        target = [
            s for s in samples if s.outcome in (SampleOutcome.PASS, SampleOutcome.PHYSICAL_FAIL)
        ]
        label = f"all_{len(target)}_evaluable"

    else:
        return [], "unknown"

    for s in target:
        _run_one_p06(s, combo, base_graph, context, library, slot_uncertainties)

    return target, label


def _run_one_p06(
    sample: SampleResult,
    combo: CatalogCombo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
    slot_uncertainties: list,
) -> None:
    """Run P06 adaptive sweep for one sample's drawn values and annotate it."""
    from foster_eom.analysis.sweep import SweepSpec, compute_adaptive_sweep
    from foster_eom.catalog.component import FallbackPolicy
    from foster_eom.realization.substitute import build_substituted_graph
    from foster_eom.robustness.evaluator import _build_perturbed_model

    su_map = {su.element_id: su for su in slot_uncertainties}
    model_overrides: dict[str, Any] = {}

    freq_range: tuple[float, float] | None = None
    if context.evaluation_frequencies_hz:
        freq_range = (
            min(context.evaluation_frequencies_hz),
            max(context.evaluation_frequencies_hz),
        )

    for element_id, entry in combo.slot_entries.items():
        su = su_map.get(element_id)
        if su is None or not su.is_stochastic:
            try:
                model = library.build_model(
                    entry.component_id, fallback=FallbackPolicy.ALLOW_LOWER_TIER
                )
                model_overrides[element_id] = model
            except Exception:
                sample.verify_passed = False
                sample.verify_report = {"error": f"model build failed: {element_id}"}
                return
        else:
            try:
                model = _build_perturbed_model(
                    element_id, sample.draw, combo, su, library, freq_range
                )
                model_overrides[element_id] = model
            except Exception:
                sample.verify_passed = False
                sample.verify_report = {"error": f"perturbed model failed: {element_id}"}
                return

    try:
        subst_graph = build_substituted_graph(base_graph, model_overrides)
    except Exception as exc:
        sample.verify_passed = False
        sample.verify_report = {"error": f"graph substitution: {exc}"}
        return

    target_hz = tuple(context.evaluation_frequencies_hz[i] for i in context.target_indices)
    if context.p06_sweep_band_hz is not None:
        f_min, f_max = context.p06_sweep_band_hz
        sweep_spec = SweepSpec(f_min_hz=f_min, f_max_hz=f_max)
    else:
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
        sample.verify_passed = passed
        sample.verify_report = {
            "verification_complete": sweep_result.verification_complete,
            "n_evaluations": len(sweep_result.frequencies_hz),
            "failed_frequencies_hz": list(failed),
            "resonance_peaks": len(sweep_result.resonance_list),
            "off_target_unsafe": sweep_result.off_target_unsafe,
        }
        # Extract resonance locations for distribution stats
        if sweep_result.resonance_list:
            sample.resonance_hz = [r.frequency_hz for r in sweep_result.resonance_list]
    except Exception as exc:
        sample.verify_passed = False
        sample.verify_report = {"error": str(exc)}
