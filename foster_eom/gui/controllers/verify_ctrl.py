"""Controller for verification."""

from __future__ import annotations

from foster_eom.analysis.q_factor import compute_q_metrics
from foster_eom.analysis.stress import compute_stress
from foster_eom.analysis.sweep import compute_adaptive_sweep
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState


class VerifyCtrl:
    @staticmethod
    def run(state: ProjectState, opt_result: object) -> tuple[object, list[object], object]:
        """Run verification suite and return (sweep_result, q_metrics, stress_summary)."""
        from foster_eom.optimize.engine import OptimizationResult

        if not isinstance(opt_result, OptimizationResult):
            raise TypeError("Expected OptimizationResult")

        if opt_result.best_feasible is None and opt_result.near_feasible_best is None:
            raise ValueError("No feasible or near-feasible solution to verify.")

        cand = opt_result.best_feasible or opt_result.near_feasible_best
        if cand is None:
            raise ValueError("Candidate is None")

        spec = state_to_spec(state)

        sweep_res = compute_adaptive_sweep(cand, spec.frequencies, spec.matching, spec.eom)
        stress_res = compute_stress(sweep_res, spec.stress)
        q_metrics = compute_q_metrics(sweep_res, spec.q_bandwidth)

        return sweep_res, q_metrics, stress_res
