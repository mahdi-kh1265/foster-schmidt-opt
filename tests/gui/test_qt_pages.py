"""Qt tests for GUI pages — controls and result rendering."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from foster_eom.gui.pages.project_page import ProjectPage, _FreqRow
from foster_eom.gui.state import ProjectState


class TestProjectPage:
    """Test ProjectPage widget logic."""

    def test_populate_and_writeback(self, qtbot):
        page = ProjectPage()
        qtbot.addWidget(page)

        state = ProjectState()
        state.name = "Test Project"
        state.frequencies_hz = [5e6, 10e6]
        state.sweep_f_min_hz = 2e6
        state.sweep_f_max_hz = 20e6
        state.source.mode = "thevenin"
        state.source.vth_rms = 2.0
        state.source.z_source_ohm = 75.0
        state.eom.model_type = "lossy_capacitor"
        state.eom.c0_f = 100e-12
        state.eom.rs_ohm = 0.5
        state.topology.n_branches = 2
        state.topology.n_cells_per_branch = 3

        page.populate_from_state(state)

        assert page.name_edit.text() == "Test Project"
        assert page.source_mode.currentText() == "thevenin"
        assert page.vth_rms.value() == pytest.approx(2.0)
        assert page.z_source.value() == pytest.approx(75.0)
        assert page.eom_type.currentText() == "lossy_capacitor"
        assert page.eom_c0.value() == pytest.approx(100.0)  # pF
        assert page.n_branches.value() == 2
        assert page.n_cells.value() == 3

        # Write back
        out = ProjectState()
        page.write_to_state(out)
        assert out.name == "Test Project"
        assert len(out.frequencies_hz) == 2
        assert out.frequencies_hz[0] == pytest.approx(5e6, rel=0.01)
        assert out.frequencies_hz[1] == pytest.approx(10e6, rel=0.01)
        assert out.source.mode == "thevenin"
        assert out.source.vth_rms == pytest.approx(2.0)
        assert out.eom.c0_f == pytest.approx(100e-12, rel=0.01)
        assert out.topology.n_branches == 2

    def test_add_remove_freq_row(self, qtbot):
        page = ProjectPage()
        qtbot.addWidget(page)
        page.populate_from_state(ProjectState())

        initial_count = page.freq_list_layout.count()
        page._add_freq_row(20e6)
        assert page.freq_list_layout.count() == initial_count + 1

    def test_validation_catches_empty_freqs(self, qtbot):
        page = ProjectPage()
        qtbot.addWidget(page)

        state = ProjectState()
        state.frequencies_hz = []
        page.populate_from_state(state)

        # Manually clear the row that populate_from_state adds by default
        while page.freq_list_layout.count():
            w = page.freq_list_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

        err = page.validate()
        assert err is not None
        assert "frequency" in err.lower()

    def test_validation_catches_bad_sweep(self, qtbot):
        page = ProjectPage()
        qtbot.addWidget(page)

        state = ProjectState()
        state.frequencies_hz = [10e6]
        page.populate_from_state(state)
        page.sweep_min.setValue(50.0)
        page.sweep_max.setValue(5.0)
        err = page.validate()
        assert err is not None
        assert "sweep" in err.lower()

    def test_freq_row_unit_conversion(self, qtbot):
        row = _FreqRow(1e9)
        qtbot.addWidget(row)
        assert row.freq_hz() == pytest.approx(1e9, rel=0.01)

    def test_validation_passes_good_inputs(self, qtbot):
        page = ProjectPage()
        qtbot.addWidget(page)

        state = ProjectState()
        state.frequencies_hz = [10e6]
        state.sweep_f_min_hz = 1e6
        state.sweep_f_max_hz = 30e6
        page.populate_from_state(state)
        err = page.validate()
        assert err is None


class TestLibraryPage:
    """Test LibraryPage widget creation."""

    def test_creates_without_crash(self, qtbot):
        from foster_eom.gui.pages.library_page import LibraryPage

        page = LibraryPage()
        qtbot.addWidget(page)
        assert page.library_path is None
        assert page.table.columnCount() == 7


class TestSynthesizePage:
    """Test SynthesizePage controls."""

    def test_run_disabled_without_state(self, qtbot):
        from foster_eom.gui.pages.synthesize_page import SynthesizePage

        page = SynthesizePage()
        qtbot.addWidget(page)
        assert not page.btn_run.isEnabled()

    def test_run_enabled_after_set_state(self, qtbot):
        from foster_eom.gui.pages.synthesize_page import SynthesizePage

        page = SynthesizePage()
        qtbot.addWidget(page)

        state = ProjectState()
        state.frequencies_hz = [10e6]
        page.set_state(state)
        assert page.btn_run.isEnabled()


class TestVerifyPage:
    """Test VerifyPage controls."""

    def test_run_disabled_without_prereqs(self, qtbot):
        from foster_eom.gui.pages.verify_page import VerifyPage

        page = VerifyPage()
        qtbot.addWidget(page)
        assert not page.btn_run.isEnabled()

    def test_run_enabled_with_prereqs(self, qtbot):
        from foster_eom.gui.pages.verify_page import VerifyPage

        page = VerifyPage()
        qtbot.addWidget(page)

        state = ProjectState()
        page.set_prereqs(state, object())  # any non-None opt result
        assert page.btn_run.isEnabled()


class TestRealizationPage:
    """Test RealizationPage controls."""

    def test_run_disabled_without_prereqs(self, qtbot):
        from foster_eom.gui.pages.realization_page import RealizationPage

        page = RealizationPage()
        qtbot.addWidget(page)
        assert not page.btn_run.isEnabled()

    def test_run_enabled_with_both_prereqs(self, qtbot):
        from foster_eom.gui.pages.realization_page import RealizationPage

        page = RealizationPage()
        qtbot.addWidget(page)

        state = ProjectState()
        state.library_path = "/fake/lib.db"
        page.set_prereqs(state, object())
        assert page.btn_run.isEnabled()

    def test_run_disabled_without_library(self, qtbot):
        from foster_eom.gui.pages.realization_page import RealizationPage

        page = RealizationPage()
        qtbot.addWidget(page)

        state = ProjectState()
        page.set_prereqs(state, object())
        assert not page.btn_run.isEnabled()


class TestRobustnessPage:
    """Test RobustnessPage controls."""

    def test_run_disabled_without_prereqs(self, qtbot):
        from foster_eom.gui.pages.robustness_page import RobustnessPage

        page = RobustnessPage()
        qtbot.addWidget(page)
        assert not page.btn_run.isEnabled()


class TestSpicePage:
    """Test SpicePage controls."""

    def test_buttons_disabled_without_prereqs(self, qtbot):
        from foster_eom.gui.pages.spice_page import SpicePage

        page = SpicePage()
        qtbot.addWidget(page)
        assert not page.btn_generate.isEnabled()
        assert not page.btn_export.isEnabled()
        assert not page.btn_validate.isEnabled()

    def test_generate_enabled_with_prereqs(self, qtbot):
        from foster_eom.gui.pages.spice_page import SpicePage

        page = SpicePage()
        qtbot.addWidget(page)

        state = ProjectState()
        page.set_prereqs(state, object())
        assert page.btn_generate.isEnabled()
        assert page.btn_validate.isEnabled()
        assert not page.btn_export.isEnabled()  # until netlist generated


class TestMainWindow:
    """Test MainWindow creation and navigation."""

    def test_window_creates_with_all_pages(self, qtbot):
        from foster_eom.gui.main_window import MainWindow

        w = MainWindow()
        qtbot.addWidget(w)
        assert w.nav_list.count() == 7
        assert w.pages.count() == 7

    def test_page_navigation(self, qtbot):
        from foster_eom.gui.main_window import MainWindow

        w = MainWindow()
        qtbot.addWidget(w)
        w.nav_list.setCurrentRow(2)
        assert w.pages.currentIndex() == 2

    def test_new_project_resets_state(self, qtbot):
        from foster_eom.gui.main_window import MainWindow

        w = MainWindow()
        qtbot.addWidget(w)
        w._new_project()
        assert w._state.name == ""
        assert len(w._state.frequencies_hz) == 1
