"""P10 RobustnessResult data classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foster_eom.realization.result import CatalogCombo
    from foster_eom.robustness.evaluator import SampleResult
    from foster_eom.robustness.sampler import RobustnessSpec
    from foster_eom.robustness.sensitivity import FailureAssociation, OATSensitivity
    from foster_eom.robustness.stats import DistributionStats, YieldStats
    from foster_eom.robustness.uncertainty import SlotUncertainty


@dataclass
class RobustnessResult:
    """Complete result of a P10 robustness analysis run.

    Parameters
    ----------
    spec : RobustnessSpec
    combo : CatalogCombo
        The frozen P09 combo analysed.
    slot_uncertainties : list[SlotUncertainty]
        Per-slot uncertainty specs (including deterministic slots with has_tol_frac=False).
    non_stochastic_slots : list[str]
        Slots held at nominal (no tolerance data declared in catalog).
    perturbation_notes : list[str]
        Approximation warnings recorded during analysis.
        E.g. "b1_L1: measured_residual first-order correction applied."
    samples : list[SampleResult]
        All N samples.
    yield_stats : YieldStats
    distributions : DistributionStats
    oat_sensitivity : list[OATSensitivity]
        Sorted by sensitivity_J descending.
    failure_association : list[FailureAssociation]
        Heuristic slot → PHYSICAL_FAIL association. Not causal attribution.
    p06_diagnostic_results : list[SampleResult]
        Subset of samples that had P06 run (verify_report populated).
    p06_diagnostic_label : str
        E.g. "worst_20_by_objective" or "all_487_evaluable".
    nominal_objective : float
    nominal_feasible : bool
    nominal_verify_passed : bool | None
        None if P06 was not run on the nominal combo.
    """

    spec: RobustnessSpec
    combo: CatalogCombo
    slot_uncertainties: list[SlotUncertainty]
    non_stochastic_slots: list[str]
    perturbation_notes: list[str]

    samples: list[SampleResult]

    yield_stats: YieldStats
    distributions: DistributionStats

    oat_sensitivity: list[OATSensitivity]
    failure_association: list[FailureAssociation]

    p06_diagnostic_results: list[SampleResult]
    p06_diagnostic_label: str

    nominal_objective: float
    nominal_feasible: bool
    nominal_verify_passed: bool | None = None
