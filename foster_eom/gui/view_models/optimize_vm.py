"""View models for optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field

from foster_eom.optimize.engine import OptimizationResult


@dataclass(frozen=True)
class CandidateRow:
    rank: int
    objective: float
    feasible: bool
    near_feasible: bool
    numerical_status: str


@dataclass(frozen=True)
class ConstraintDisplayRow:
    """One constraint margin for GUI display."""

    label: str  # human-readable physical label
    margin: float  # normalized margin (negative = violated)


@dataclass(frozen=True)
class CandidateDetailVM:
    """Structured candidate detail for the GUI panel."""

    rank: int
    topology_id: str
    objective_base: float
    objective_soft: float
    feasible: bool
    near_feasible: bool
    v_max: float
    numerical_status: str
    local_polish_method: str
    local_polish_outcome: str
    seed_source: str
    objective_terms: dict[str, float] = field(default_factory=dict)

    # Constraint summary
    total_hard: int = 0
    violated_count: int = 0
    violated: list[ConstraintDisplayRow] = field(default_factory=list)
    closest_active: list[ConstraintDisplayRow] = field(default_factory=list)
    all_constraints: list[ConstraintDisplayRow] = field(default_factory=list)

    @classmethod
    def from_candidate(
        cls,
        rank: int,
        c: object,
        label_map: dict[str, str] | None = None,
    ) -> CandidateDetailVM:
        """Build a CandidateDetailVM from a CandidateResult.

        Parameters
        ----------
        rank : int
            1-based rank of this candidate.
        c : CandidateResult
            The candidate result (typed as object to avoid circular import
            at module level; actual type is CandidateResult).
        label_map : dict[str, str], optional
            Mapping of canonical margin keys (e.g., 'hard_0') to human labels.
        """
        from foster_eom.domain.results import CandidateResult

        assert isinstance(c, CandidateResult)

        margins = c.constraint_margins
        total_hard = len(margins)

        all_rows = [
            ConstraintDisplayRow(
                label=label_map.get(k, k) if label_map else k,
                margin=v
            ) for k, v in margins.items()
        ]

        # Violated: negative margins, sorted worst-first (most negative first)
        violated = sorted(
            [r for r in all_rows if r.margin < 0.0],
            key=lambda r: r.margin,
        )

        # Closest active: nonneg margins, sorted smallest-first, top ~10
        nonneg = sorted(
            [r for r in all_rows if r.margin >= 0.0],
            key=lambda r: r.margin,
        )
        closest_active = nonneg[:10]

        # Polish outcome display
        method = c.local_polish_method or ""
        outcome = c.local_polish_outcome or ""

        return cls(
            rank=rank,
            topology_id=c.topology_id,
            objective_base=c.base_objective_value,
            objective_soft=c.soft_penalty_total,
            feasible=c.feasible,
            near_feasible=c.near_feasible,
            v_max=c.v_max,
            numerical_status=c.numerical_status,
            local_polish_method=method,
            local_polish_outcome=outcome,
            seed_source=c.seed_source,
            objective_terms=dict(c.objective_terms),
            total_hard=total_hard,
            violated_count=len(violated),
            violated=violated,
            closest_active=closest_active,
            all_constraints=all_rows,
        )


_OUTCOME_DISPLAY: dict[str, tuple[str, str]] = {
    "polished_retained": ("polished candidate retained", ""),
    "pre_polish_retained": ("pre-polish candidate retained (Deb-better)", ""),
    "fd_fallback": ("REFERENCE_FD fallback", ""),
    "not_selected": ("not selected for local polish", ""),
}


def format_polish_provenance(method: str, outcome: str) -> list[str]:
    """Format local-polish provenance for display.

    Returns a list of human-readable lines.
    """
    lines: list[str] = []

    if not outcome and not method:
        lines.append("Local polish: (no provenance)")
        return lines

    if outcome == "not_selected":
        lines.append("Local polish: not selected for local polish")
        return lines

    if outcome == "fd_fallback":
        lines.append(f"Local polish: REFERENCE_FD fallback / {method}" if method else "Local polish: REFERENCE_FD fallback")
        return lines

    # ANALYTICAL cases
    mode_str = "ANALYTICAL" if outcome in ("polished_retained", "pre_polish_retained") else ""
    if method:
        lines.append(f"Local polish: {mode_str} / {method}" if mode_str else f"Local polish: {method}")
    elif mode_str:
        lines.append(f"Local polish: {mode_str}")

    display = _OUTCOME_DISPLAY.get(outcome)
    if display:
        lines.append(f"Outcome: {display[0]}")
    elif outcome:
        lines.append(f"Outcome: {outcome}")

    return lines


@dataclass(frozen=True)
class OptimizeVM:
    status_label: str
    candidates: list[CandidateRow]

    @classmethod
    def from_result(cls, r: OptimizationResult) -> OptimizeVM:
        if r.best_feasible is not None:
            status = "FEASIBLE"
        elif r.near_feasible_best is not None:
            status = "NEAR-FEASIBLE"
        else:
            status = "INFEASIBLE"

        rows = []
        for i, c in enumerate(r.candidates):
            rows.append(
                CandidateRow(
                    rank=i + 1,
                    objective=c.base_objective_value + c.soft_penalty_total,
                    feasible=c.feasible,
                    near_feasible=c.near_feasible,
                    numerical_status=c.numerical_status,
                )
            )

        return cls(status_label=status, candidates=rows)
