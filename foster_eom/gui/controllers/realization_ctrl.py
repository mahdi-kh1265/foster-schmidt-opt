"""Controller for catalog realization."""

from __future__ import annotations

from foster_eom.catalog.library import ComponentLibrary
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState
from foster_eom.realization.runner import realize
from foster_eom.realization.spec import RealizationSpec


class RealizationCtrl:
    @staticmethod
    def run(state: ProjectState, opt_result: object) -> object:
        """Run realization using a fresh library connection."""
        from foster_eom.optimize.engine import OptimizationResult

        if not isinstance(opt_result, OptimizationResult):
            raise TypeError("Expected OptimizationResult")

        if not state.library_path:
            raise ValueError("No library selected.")

        if opt_result.best_feasible is None and opt_result.near_feasible_best is None:
            raise ValueError("No feasible or near-feasible solution to realize.")

        eval_res = opt_result.best_feasible or opt_result.near_feasible_best
        if eval_res is None:
            raise ValueError("Candidate is None")

        project_spec = state_to_spec(state)
        real_spec = RealizationSpec(
            project_spec=project_spec,
            baseline_result=eval_res,
        )

        lib = ComponentLibrary(state.library_path)
        try:
            return realize(real_spec, lib)
        finally:
            lib.close()
