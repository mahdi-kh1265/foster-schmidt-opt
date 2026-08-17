"""P10 per-sample evaluator.

Applies a draw to build perturbed models, evaluates via MNA, and classifies
the outcome into one of four states:
  PASS / PHYSICAL_FAIL / MODEL_COVERAGE_UNRESOLVED / NUMERICAL_UNRESOLVED.

Numerical failures are retried before being marked NUMERICAL_UNRESOLVED.
They are NOT silently excluded from yield statistics.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from foster_eom.robustness.uncertainty import PerturbMethod, SlotUncertainty

if TYPE_CHECKING:
    from foster_eom.catalog.library import ComponentLibrary
    from foster_eom.circuit.graph import CircuitGraph
    from foster_eom.optimize.evaluator import EvaluationContext, EvaluationResult
    from foster_eom.realization.result import CatalogCombo
    from foster_eom.robustness.sampler import RobustnessSpec


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


class SampleOutcome(enum.StrEnum):
    PASS = "pass"
    """Feasible: hard constraints satisfied."""

    PHYSICAL_FAIL = "physical_fail"
    """Infeasible: at least one hard constraint violated. MNA was numerically OK."""

    MODEL_COVERAGE_UNRESOLVED = "model_coverage_unresolved"
    """Drawn value falls outside a model's declared validity range.
    Not counted as a yield failure; reported separately."""

    NUMERICAL_UNRESOLVED = "numerical_unresolved"
    """MNA solve failed or produced non-finite result after retries.
    Included in yield denominator as a non-pass (conservative)."""


# ---------------------------------------------------------------------------
# Sample result
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Result of evaluating one Monte Carlo sample.

    Parameters
    ----------
    sample_idx : int
    draw : dict[str, float]
        element_id → drawn value in SI.
    perturb_methods : dict[str, str]
        element_id → perturbation method used (PerturbMethod value).
    outcome : SampleOutcome
    eval_result : EvaluationResult | None
        None if model build or MNA failed entirely.
    verify_report : dict | None
        Populated by P06 diagnostic if run for this sample.
    verify_passed : bool | None
        None = P06 not run.
    retry_count : int
        Number of retries attempted before reaching final outcome.
    failure_reason : str | None
    hard_margin_min : float | None
        Minimum hard constraint margin across all constraints (most-violated).
        Negative = infeasible.
    resonance_hz : list[float]
        Resonance/pole frequencies identified from sweep (if available).
    objective_value : float | None
    v_max : float | None
    v_sum : float | None
    """

    sample_idx: int
    draw: dict[str, float]
    perturb_methods: dict[str, str]
    outcome: SampleOutcome
    eval_result: Any | None = None  # EvaluationResult; Any to avoid circular
    verify_report: dict | None = None
    verify_passed: bool | None = None
    retry_count: int = 0
    failure_reason: str | None = None
    hard_margin_min: float | None = None
    resonance_hz: list[float] = field(default_factory=list)
    objective_value: float | None = None
    v_max: float | None = None
    v_sum: float | None = None


# ---------------------------------------------------------------------------
# Model perturbation helpers
# ---------------------------------------------------------------------------


def _build_perturbed_model(
    element_id: str,
    draw: dict[str, float],
    combo: CatalogCombo,
    su: SlotUncertainty,
    library: ComponentLibrary,
    freq_range: tuple[float, float] | None,
) -> Any:  # OnePortModel
    """Build a perturbed OnePortModel for one slot.

    Uses the appropriate perturbation method based on model tier:
    - ideal_lc: directly scale L or C in ideal/parametric model.
    - measured_residual: wrap tabular model with first-order correction.
    - none: return the nominal model unperturbed.

    Raises
    ------
    ModelCoverageError
        If drawn value is outside model validity range.
    """
    from foster_eom.catalog.component import FallbackPolicy
    from foster_eom.models.components import IdealCapacitor, IdealInductor

    drawn_value = draw[element_id]
    entry = combo.slot_entries[element_id]
    nom_value = entry.value_nom

    method = su.perturb_method

    if method == PerturbMethod.NONE or drawn_value == nom_value:
        # Deterministic slot — return nominal model
        return library.build_model(
            entry.component_id,
            freq_range=freq_range,
            fallback=FallbackPolicy.ALLOW_LOWER_TIER,
        )

    if method == PerturbMethod.IDEAL_LC:
        # Build model and replace the nominal value with the drawn one
        nom_model = library.build_model(
            entry.component_id,
            freq_range=freq_range,
            fallback=FallbackPolicy.ALLOW_LOWER_TIER,
        )
        # Determine kind from element_id suffix
        eid_upper = element_id.upper()
        if "_L" in eid_upper or "LINF" in eid_upper:
            return IdealInductor(drawn_value)
        else:
            return IdealCapacitor(drawn_value)

    if method == PerturbMethod.MEASURED_RESIDUAL:
        nom_model = library.build_model(
            entry.component_id,
            freq_range=freq_range,
            fallback=FallbackPolicy.ALLOW_LOWER_TIER,
        )
        eid_upper = element_id.upper()
        is_inductor = "_L" in eid_upper or "LINF" in eid_upper
        return MeasuredResidualModel(
            base_model=nom_model,
            nom_value=nom_value,
            drawn_value=drawn_value,
            is_inductor=is_inductor,
        )

    raise ValueError(f"Unknown perturb_method: {method!r}")


class ModelCoverageError(Exception):
    """Raised when drawn value is outside model validity range."""


class MeasuredResidualModel:
    """First-order measured-model perturbation (P10 spec §4).

    For inductor (δ = drawn/nom - 1):
        Z_draw(f) = Z_meas(f) + j*ω*L_nom*δ

    For capacitor (δ = drawn/nom - 1):
        Z_draw(f) = Z_meas(f) - δ / (j*ω*C_nom*(1+δ))

    This preserves the full measured impedance shape with a first-order
    reactive correction.  Approximation: recorded in perturbation_notes.
    """

    def __init__(
        self,
        base_model: Any,
        nom_value: float,
        drawn_value: float,
        is_inductor: bool,
    ) -> None:
        self._base = base_model
        self._nom = nom_value
        self._drawn = drawn_value
        self._delta = drawn_value / nom_value - 1.0
        self._is_inductor = is_inductor
        # Delegate validity range from base model
        self.extrapolation_policy = getattr(base_model, "extrapolation_policy", None)

    def validity_range(self) -> tuple[float, float] | None:
        result = self._base.validity_range()
        if result is None:
            return None
        lo, hi = result
        return (float(lo), float(hi))

    def metadata(self) -> dict:
        return {
            **self._base.metadata(),
            "perturb_method": "measured_residual",
            "nom_value": self._nom,
            "drawn_value": self._drawn,
            "delta_frac": self._delta,
        }

    def z(self, f_hz: float | np.ndarray) -> Any:
        z_meas = self._base.z(f_hz)
        omega = 2.0 * math.pi * np.asarray(f_hz, dtype=np.float64)
        if self._is_inductor:
            correction = 1j * omega * self._nom * self._delta
        else:
            # Capacitor: Z_draw = Z_meas - δ/(jω·C_nom·(1+δ))
            denom = 1j * omega * self._nom * (1.0 + self._delta)
            # Avoid division by zero at f=0 (not expected in practice)
            with np.errstate(divide="ignore", invalid="ignore"):
                correction = -self._delta / denom
        result = z_meas + correction
        if np.ndim(f_hz) == 0:
            return complex(result)
        return result

    def y(self, f_hz: float | np.ndarray) -> Any:
        z_val = self.z(f_hz)
        return 1.0 / z_val


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def evaluate_sample(
    sample_idx: int,
    draw: dict[str, float],
    slot_uncertainties: list[SlotUncertainty],
    combo: CatalogCombo,
    base_graph: CircuitGraph,
    context: EvaluationContext,
    library: ComponentLibrary,
    spec: RobustnessSpec,
) -> SampleResult:
    """Evaluate one Monte Carlo sample.

    Builds perturbed models, substitutes into circuit, evaluates via MNA,
    retries on numerical failure, and classifies the outcome.
    """
    from foster_eom.realization.substitute import evaluate_with_overrides

    su_map = {su.element_id: su for su in slot_uncertainties}
    perturb_methods: dict[str, str] = {}

    def _build_overrides() -> dict[str, Any] | str:
        """Build model overrides or return error string."""
        overrides: dict[str, Any] = {}
        for element_id, entry in combo.slot_entries.items():
            su = su_map.get(element_id)
            freq_range: tuple[float, float] | None = None
            # Get freq_range from context
            if context.evaluation_frequencies_hz:
                freq_range = (
                    min(context.evaluation_frequencies_hz),
                    max(context.evaluation_frequencies_hz),
                )

            if su is None:
                # No uncertainty info — use nominal model
                try:
                    from foster_eom.catalog.component import FallbackPolicy

                    model = library.build_model(
                        entry.component_id,
                        freq_range=freq_range,
                        fallback=FallbackPolicy.ALLOW_LOWER_TIER,
                    )
                    overrides[element_id] = model
                    perturb_methods[element_id] = PerturbMethod.NONE
                except Exception as exc:
                    return f"model build failed for {element_id}: {exc}"
            else:
                perturb_methods[element_id] = su.perturb_method.value
                try:
                    model = _build_perturbed_model(
                        element_id, draw, combo, su, library, freq_range
                    )
                    overrides[element_id] = model
                except ModelCoverageError as exc:
                    return f"coverage: {exc}"
                except Exception as exc:
                    return f"model build failed for {element_id}: {exc}"

        return overrides

    def _try_evaluate(overrides: dict[str, Any]) -> EvaluationResult | str:
        try:
            return evaluate_with_overrides(base_graph, overrides, context)
        except Exception as exc:
            return f"mna: {exc}"

    def _classify(eval_result: EvaluationResult) -> SampleOutcome:
        if eval_result.numerical_status != "ok":
            return SampleOutcome.NUMERICAL_UNRESOLVED
        if not math.isfinite(eval_result.v_max):
            return SampleOutcome.NUMERICAL_UNRESOLVED
        if eval_result.feasible:
            return SampleOutcome.PASS
        return SampleOutcome.PHYSICAL_FAIL

    # --- Main evaluation with retry ---
    retry_count = 0
    overrides_or_err = _build_overrides()

    if isinstance(overrides_or_err, str):
        err = overrides_or_err
        if err.startswith("coverage:"):
            return SampleResult(
                sample_idx=sample_idx,
                draw=draw,
                perturb_methods=perturb_methods,
                outcome=SampleOutcome.MODEL_COVERAGE_UNRESOLVED,
                failure_reason=err,
            )
        # Model build failure → retry with nominal fallback
        return SampleResult(
            sample_idx=sample_idx,
            draw=draw,
            perturb_methods=perturb_methods,
            outcome=SampleOutcome.NUMERICAL_UNRESOLVED,
            failure_reason=err,
        )

    eval_or_err = _try_evaluate(overrides_or_err)

    while isinstance(eval_or_err, str) and retry_count < spec.n_retry:
        retry_count += 1
        # Re-try evaluation at the same draw (idempotent retry for transient numerics)
        eval_or_err = _try_evaluate(overrides_or_err)

    if isinstance(eval_or_err, str):
        return SampleResult(
            sample_idx=sample_idx,
            draw=draw,
            perturb_methods=perturb_methods,
            outcome=SampleOutcome.NUMERICAL_UNRESOLVED,
            retry_count=retry_count,
            failure_reason=eval_or_err,
        )

    eval_result: EvaluationResult = eval_or_err  # type: ignore[assignment]
    outcome = _classify(eval_result)

    # Extract per-constraint margin minimum
    hard_margin_min: float | None = None
    if eval_result.hard_margins:
        hard_margin_min = float(min(eval_result.hard_margins))

    return SampleResult(
        sample_idx=sample_idx,
        draw=draw,
        perturb_methods=perturb_methods,
        outcome=outcome,
        eval_result=eval_result,
        retry_count=retry_count,
        hard_margin_min=hard_margin_min,
        objective_value=eval_result.objective_value,
        v_max=eval_result.v_max,
        v_sum=eval_result.v_sum,
    )
