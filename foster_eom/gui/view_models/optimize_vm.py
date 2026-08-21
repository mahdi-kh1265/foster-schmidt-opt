"""View models for optimization results."""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.optimize.engine import OptimizationResult


@dataclass(frozen=True)
class CandidateRow:
    rank: int
    objective: float
    feasible: bool
    near_feasible: bool
    numerical_status: str


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
