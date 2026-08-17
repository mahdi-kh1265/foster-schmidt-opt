"""Controller for project state loading/saving."""

from __future__ import annotations

from pathlib import Path

from foster_eom.gui.adapter import load_gui_project, save_gui_project
from foster_eom.gui.state import ProjectState


class ProjectCtrl:
    @staticmethod
    def save(state: ProjectState, path: str | Path) -> None:
        """Save the project state and reset the modified flag."""
        save_gui_project(state, path)
        state.modified = False

    @staticmethod
    def load(path: str | Path) -> ProjectState:
        """Load the project state."""
        state = load_gui_project(path)
        state.modified = False
        return state
