"""Controller for robustness."""

from __future__ import annotations

from foster_eom.catalog.library import ComponentLibrary
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState
from foster_eom.robustness.runner import run_robustness


class RobustnessCtrl:
    @staticmethod
    def run(state: ProjectState, realization_result: object) -> object:
        """Run robustness using a fresh library connection."""
        from foster_eom.realization.result import RealizationResult

        if not isinstance(realization_result, RealizationResult):
            raise TypeError("Expected RealizationResult")

        if not state.library_path:
            raise ValueError("No library selected.")

        if realization_result.best is None:
            raise ValueError("No catalog realization available for robustness analysis.")

        spec = state_to_spec(state)

        lib = ComponentLibrary(state.library_path)
        try:
            return run_robustness(
                combo=realization_result.best,
                project_spec=spec,
                library=lib,
            )
        finally:
            lib.close()
