"""Result types for discrete catalog realization (Prompt 09)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foster_eom.optimize.evaluator import EvaluationResult
from foster_eom.realization.spec import NeighborhoodEntry

# ---------------------------------------------------------------------------
# One fully evaluated catalog combination
# ---------------------------------------------------------------------------


@dataclass
class CatalogCombo:
    """One fully evaluated discrete realization candidate.

    ``slot_entries`` maps element_id → the frozen NeighborhoodEntry selected.
    ``eval_result``  is the full MNA EvaluationResult with real component models.
    ``deb_key``      is the P05 Deb comparison tuple (lower = better).
    ``verify_report`` is populated by P06 verification (None if not yet run).
    """

    slot_entries: dict[str, NeighborhoodEntry]
    eval_result: EvaluationResult
    deb_key: tuple  # (not feasible, v_max, v_sum, objective_value)
    verify_report: dict[str, Any] | None = None
    verify_passed: bool | None = None  # None = not yet verified


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class RealizationDiagnostics:
    """Search accounting for one realization run."""

    n_slots: int
    parts_per_slot: dict[str, int]  # element_id → count
    total_combos: int  # product of per-slot counts
    n_combos_generated: int
    n_combos_evaluated: int
    n_mna_solves: int
    search_exhaustive: bool
    search_truncated: bool
    budget_exhausted: bool


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class RealizationResult:
    """Complete result of a discrete catalog realization run.

    Parameters
    ----------
    status : str
        ``"feasible"`` — at least one combo passes P06 gates.
        ``"degraded"`` — no combo fully feasible; best is near-feasible.
        ``"infeasible"`` — exhaustive search; all combos fail hard constraints.
        ``"no_feasible_found"`` — beam search; no feasible combo found;
            global infeasibility NOT claimed.
        ``"no_candidates"`` — catalog returned zero parts for >=1 slot.
    continuous_baseline : EvaluationResult
        The original P05 continuous evaluation result.
    combos : list[CatalogCombo]
        All evaluated combos, Deb-sorted (best first).
    best : CatalogCombo | None
        The Deb-best combo, or None if no combos evaluated.
    degradation : float | None
        ``best.eval_result.objective_value - continuous_baseline.objective_value``.
        Positive = degraded, negative = improved. None if no combos.
    failed_slots : list[str]
        Element IDs with zero catalog candidates.
    diagnostics : RealizationDiagnostics
    verified_combos : list[CatalogCombo]
        Subset of combos for which P06 verify was attempted (in Deb order).
    first_passing_combo : CatalogCombo | None
        First combo whose P06 verify passed, or None.
    """

    status: str
    continuous_baseline: EvaluationResult
    combos: list[CatalogCombo] = field(default_factory=list)
    best: CatalogCombo | None = None
    degradation: float | None = None
    failed_slots: list[str] = field(default_factory=list)
    diagnostics: RealizationDiagnostics | None = None
    verified_combos: list[CatalogCombo] = field(default_factory=list)
    first_passing_combo: CatalogCombo | None = None
