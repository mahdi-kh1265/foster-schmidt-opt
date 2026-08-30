"""Main window — wires pages, New/Open/Save, downstream gating, stale detection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from foster_eom.gui.adapter import load_gui_project, save_gui_project
from foster_eom.gui.pages.library_page import LibraryPage
from foster_eom.gui.pages.project_page import ProjectPage
from foster_eom.gui.pages.realization_page import RealizationPage
from foster_eom.gui.pages.robustness_page import RobustnessPage
from foster_eom.gui.pages.spice_page import SpicePage
from foster_eom.gui.pages.synthesize_page import SynthesizePage
from foster_eom.gui.pages.verify_page import VerifyPage
from foster_eom.gui.state import ProjectState


class MainWindow(QMainWindow):
    """Top-level window for Foster-Schmidt Optimizer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Foster-Schmidt Optimizer")
        self.resize(1100, 750)

        self._state = ProjectState()
        self._project_path: str | None = None

        self._build_menus()
        self._build_ui()
        self._connect_signals()
        self._new_project()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        act_new = QAction("&New Project", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save = QAction("&Save Project", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save &As…", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Navigation
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(170)
        self.nav_list.setStyleSheet(
            "QListWidget { font-size: 12px; }"
            "QListWidget::item { padding: 6px 8px; }"
            "QListWidget::item:selected { background: #3377bb; color: white; }"
        )

        self.pages = QStackedWidget()

        # --- Pages ---
        self.project_page = ProjectPage()
        self.library_page = LibraryPage()
        self.synthesize_page = SynthesizePage()
        self.verify_page = VerifyPage()
        self.realization_page = RealizationPage()
        self.robustness_page = RobustnessPage()
        self.spice_page = SpicePage()

        page_defs = [
            ("1. Project", self.project_page),
            ("2. Library", self.library_page),
            ("3. Synthesize", self.synthesize_page),
            ("4. Verify", self.verify_page),
            ("5. Realization", self.realization_page),
            ("6. Robustness", self.robustness_page),
            ("7. SPICE Export", self.spice_page),
        ]
        for label, page in page_defs:
            self.nav_list.addItem(label)
            self.pages.addWidget(page)

        self.nav_list.currentRowChanged.connect(self._on_page_change)
        self.nav_list.setCurrentRow(0)

        layout.addWidget(self.nav_list)
        layout.addWidget(self.pages, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self) -> None:
        self.synthesize_page.btn_run.clicked.connect(self._on_inputs_committed)
        self.synthesize_page.optimization_finished.connect(self._on_optimization_finished)
        self.library_page.library_changed.connect(self._on_library_changed)


    def _on_library_changed(self, lib_path: str) -> None:
        if not self._state:
            return
        self._state.library_path = lib_path
        self._state.library_sha = self.library_page.library_sha
        self._state.invalidate_library()
        self._update_downstream()

    def _on_optimization_finished(self, _result: object) -> None:
        """When optimization finishes, push the new result to downstream pages."""
        self._update_downstream()

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------
    def _new_project(self) -> None:
        self._state = ProjectState()
        self._project_path = None
        self._state.frequencies_hz = [10e6]
        self.project_page.populate_from_state(self._state)
        self._update_downstream()
        self.setWindowTitle("Foster-Schmidt Optimizer — Untitled")
        self.status_bar.showMessage("New project created")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "FSEOM Project (*.fseom.yaml *.yaml);;All (*)",
        )
        if not path:
            return
        try:
            self._state = load_gui_project(path)
            self._project_path = path
            self.project_page.populate_from_state(self._state)
            if self._state.library_path:
                lib_path = Path(self._state.library_path)
                if not lib_path.exists():
                    msg = QMessageBox.question(self, "Library Not Found", f"Referenced library not found:\n{lib_path}\n\nWould you like to locate it?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if msg == QMessageBox.StandardButton.Yes:
                        new_path, _ = QFileDialog.getOpenFileName(self, "Locate Component Library", "", "FSEOM Library (*.fseom.db *.db);;All (*)")
                        if new_path:
                            self._state.library_path = new_path
                            lib_path = Path(new_path)

                if lib_path.exists():
                    self.library_page.set_library(self._state.library_path)

                    if self._state.library_sha and self.library_page.library_sha != self._state.library_sha:
                        reply = QMessageBox.question(
                            self,
                            "Library SHA Mismatch",
                            f"The library at {lib_path} has changed since this project was last saved.\n\n"
                            f"Expected: {self._state.library_sha}\n"
                            f"Found: {self.library_page.library_sha}\n\n"
                            "Do you want to permanently update the project to use this new library version? "
                            "(If No, the project remains in a mismatched state, which may invalidate downstream artifacts)",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self._state.library_sha = self.library_page.library_sha
                            self._state.invalidate_library()
                    else:
                        self._state.library_sha = self.library_page.library_sha
            self._update_downstream()
            self.setWindowTitle(f"Foster-Schmidt Optimizer — {Path(path).stem}")
            self.status_bar.showMessage(f"Loaded {path}")
        except Exception as e:
            QMessageBox.warning(self, "Open Error", str(e))

    def _save_project(self) -> None:
        if self._project_path:
            self._do_save(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "FSEOM Project (*.fseom.yaml);;All (*)",
        )
        if path:
            self._project_path = path
            self._do_save(path)

    def _do_save(self, path: str) -> None:
        self.project_page.write_to_state(self._state)
        self._state.library_path = self.library_page.library_path
        self._state.library_sha = self.library_page.library_sha
        try:
            save_gui_project(self._state, path)
            self._state.modified = False
            self.setWindowTitle(f"Foster-Schmidt Optimizer — {Path(path).stem}")
            self.status_bar.showMessage(f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    # ------------------------------------------------------------------
    # Page change + downstream gating
    # ------------------------------------------------------------------
    def _on_page_change(self, idx: int) -> None:
        # Commit project inputs when leaving page 0
        if self.pages.currentIndex() == 0 and idx != 0:
            self._on_inputs_committed()
        self.pages.setCurrentIndex(idx)

    def _on_inputs_committed(self) -> None:
        """Read current project page inputs into state, update downstream."""
        err = self.project_page.validate()
        if err:
            self.project_page.validation_label.setText(err)
            self.project_page.validation_label.setStyleSheet("color: red;")
            return
        self.project_page.validation_label.setText("")

        self.project_page.write_to_state(self._state)
        self._state.library_path = self.library_page.library_path
        self._state.library_sha = self.library_page.library_sha
        self._state.bump_revision()
        self._update_downstream()

    def _update_downstream(self) -> None:
        """Push current state + results into downstream pages."""
        self.synthesize_page.set_state(self._state)

        opt_res = self.synthesize_page.result
        self.verify_page.set_prereqs(self._state, opt_res)
        self.realization_page.set_prereqs(self._state, opt_res)

        real_res = self.realization_page.result
        self.robustness_page.set_prereqs(self._state, real_res)
        self.spice_page.set_prereqs(self._state, real_res)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._state.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "The project has unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
