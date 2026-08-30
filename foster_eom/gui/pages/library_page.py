"""Library management page."""

from __future__ import annotations

import contextlib
from pathlib import Path

from PySide6.QtCore import QItemSelection, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from foster_eom.catalog.component import ComponentKind, ModelTier
from foster_eom.catalog.query import ComponentQuery
from foster_eom.catalog.vendor_pack import VendorPackSpec
from foster_eom.gui.controllers.library_ctrl import LibraryCtrl
from foster_eom.gui.view_models.library_vm import LibraryStats


class ImportVendorPackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Vendor Pack")
        self.setMinimumWidth(400)

        lay = QVBoxLayout(self)

        form = QFormLayout()

        self.vendor_input = QLineEdit()
        self.vendor_input.setPlaceholderText("e.g. Coilcraft, Murata")
        form.addRow("Vendor Name:", self.vendor_input)

        self.adapter_combo = QComboBox()
        from foster_eom.catalog.vendor_pack import _ADAPTER_VERSIONS
        self.adapter_combo.addItems(sorted(_ADAPTER_VERSIONS.keys()))
        form.addRow("Adapter Format:", self.adapter_combo)

        # Path selection
        path_lay = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        path_lay.addWidget(self.path_input)
        path_lay.addWidget(btn_browse)
        form.addRow("Source Path:", path_lay)

        self.glob_input = QLineEdit()
        self.glob_input.setText("**/*.*")
        form.addRow("Glob Pattern:", self.glob_input)

        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self.source_path = ""

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vendor Pack",
            "",
            "ZIP archives (*.zip);;Directories (*);;All (*)",
        )
        if path:
            self.source_path = path
            self.path_input.setText(path)

class LibraryPage(QWidget):
    """Component library management page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lib_path: str | None = None
        self._stats: LibraryStats | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)

        # --- Toolbar ---
        tb = QHBoxLayout()
        self.btn_open = QPushButton("Open Library…")
        self.btn_open.clicked.connect(self._open_library)
        self.btn_create = QPushButton("New Library…")
        self.btn_create.clicked.connect(self._create_library)
        self.btn_demo = QPushButton("Create Demo Library…")
        self.btn_demo.clicked.connect(self._create_demo_library)
        self.btn_import = QPushButton("Import Vendor Pack…")
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self._import_pack)
        tb.addWidget(self.btn_open)
        tb.addWidget(self.btn_create)
        tb.addWidget(self.btn_demo)
        tb.addWidget(self.btn_import)
        tb.addStretch()
        lay.addLayout(tb)

        # --- Stats ---
        stats_grp = QGroupBox("Library Statistics")
        stats_lay = QVBoxLayout(stats_grp)
        self.lbl_path = QLabel("No library loaded")
        self.lbl_counts = QLabel("")
        self.lbl_sha = QLabel("")
        self.lbl_sha.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        stats_lay.addWidget(self.lbl_path)
        stats_lay.addWidget(self.lbl_counts)
        stats_lay.addWidget(self.lbl_sha)
        lay.addWidget(stats_grp)

        # --- Browser (Filters) ---
        filter_grp = QGroupBox("Filters")
        filter_lay = QHBoxLayout(filter_grp)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search part number...")
        filter_lay.addWidget(QLabel("Search:"))
        filter_lay.addWidget(self.search_input)

        self.vendor_combo = QComboBox()
        self.vendor_combo.addItems(["All", "POSM-DEMO"])
        filter_lay.addWidget(QLabel("Vendor:"))
        filter_lay.addWidget(self.vendor_combo)

        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["All", "Inductor", "Capacitor"])
        filter_lay.addWidget(QLabel("Kind:"))
        filter_lay.addWidget(self.kind_combo)

        self.val_min_input = QLineEdit()
        self.val_min_input.setPlaceholderText("min (e.g. 1e-12)")
        self.val_max_input = QLineEdit()
        self.val_max_input.setPlaceholderText("max")
        filter_lay.addWidget(QLabel("Value:"))
        filter_lay.addWidget(self.val_min_input)
        filter_lay.addWidget(QLabel("-"))
        filter_lay.addWidget(self.val_max_input)

        self.tier_combo = QComboBox()
        self.tier_combo.addItems(["All", "measured", "parametric", "ideal"])
        filter_lay.addWidget(QLabel("Min Tier:"))
        filter_lay.addWidget(self.tier_combo)

        self.freq_min_input = QLineEdit()
        self.freq_min_input.setPlaceholderText("min Hz")
        self.freq_max_input = QLineEdit()
        self.freq_max_input.setPlaceholderText("max Hz")
        filter_lay.addWidget(QLabel("Freq:"))
        filter_lay.addWidget(self.freq_min_input)
        filter_lay.addWidget(QLabel("-"))
        filter_lay.addWidget(self.freq_max_input)

        self.btn_apply_filters = QPushButton("Apply Filters")
        self.btn_apply_filters.clicked.connect(self._apply_filters)
        self.btn_clear_filters = QPushButton("Clear Filters")
        self.btn_clear_filters.clicked.connect(self._clear_filters)
        filter_lay.addWidget(self.btn_apply_filters)
        filter_lay.addWidget(self.btn_clear_filters)
        lay.addWidget(filter_grp)

        # --- Splitter for Table and Details ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(splitter, 1)

        # --- Part table ---
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                "Vendor",
                "Part Number",
                "Kind",
                "Nominal Value",
                "Model Tier",
                "Freq Validity",
                "Model Origin",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        # --- Details Pane ---
        self.details_grp = QGroupBox("Component Details")
        details_lay = QFormLayout(self.details_grp)
        self.lbl_det_vendor = QLabel("N/A")
        self.lbl_det_part = QLabel("N/A")
        self.lbl_det_kind = QLabel("N/A")
        self.lbl_det_pkg = QLabel("N/A")
        self.lbl_det_val = QLabel("N/A")
        self.lbl_det_tol = QLabel("N/A")
        self.lbl_det_esr = QLabel("N/A")
        self.lbl_det_srf = QLabel("N/A")
        self.lbl_det_vrating = QLabel("N/A")
        self.lbl_det_irating = QLabel("N/A")
        self.lbl_det_validity = QLabel("N/A")
        self.lbl_det_tier = QLabel("N/A")
        self.lbl_det_origin = QLabel("N/A")
        self.lbl_det_sha = QLabel("N/A")
        self.lbl_det_warning = QLabel("")
        self.lbl_det_warning.setStyleSheet("color: red; font-weight: bold;")
        self.lbl_det_warning.setWordWrap(True)

        details_lay.addRow(self.lbl_det_warning)
        details_lay.addRow("Vendor:", self.lbl_det_vendor)
        details_lay.addRow("Part Number:", self.lbl_det_part)
        details_lay.addRow("Kind:", self.lbl_det_kind)
        details_lay.addRow("Package/Series:", self.lbl_det_pkg)
        details_lay.addRow("Nominal Value:", self.lbl_det_val)
        details_lay.addRow("Tolerance:", self.lbl_det_tol)
        details_lay.addRow("DCR/ESR:", self.lbl_det_esr)
        details_lay.addRow("SRF:", self.lbl_det_srf)
        details_lay.addRow("Voltage Rating:", self.lbl_det_vrating)
        details_lay.addRow("Current Rating:", self.lbl_det_irating)
        details_lay.addRow("Freq Validity:", self.lbl_det_validity)
        details_lay.addRow("Model Tier:", self.lbl_det_tier)
        details_lay.addRow("Model Origin:", self.lbl_det_origin)
        details_lay.addRow("Model SHA256:", self.lbl_det_sha)
        splitter.addWidget(self.details_grp)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

    def _open_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Component Library",
            "",
            "FSEOM Library (*.fseom.db *.db);;All (*)",
        )
        if path:
            self._load(path)

    def _create_library(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Create Component Library",
            "",
            "FSEOM Library (*.fseom.db);;All (*)",
        )
        if path:
            from foster_eom.catalog.library import ComponentLibrary
            lib = ComponentLibrary(path)
            lib.close()
            self._load(path)

    def _create_demo_library(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Create Demo Component Library",
            "demo.fseom.db",
            "FSEOM Library (*.fseom.db);;All (*)",
        )
        if path:
            # 1. Create DB
            from foster_eom.catalog.library import ComponentLibrary
            lib = ComponentLibrary(path)
            lib.close()

            # 2. Import demo pack via standard API
            demo_pack_path = Path(__file__).parent.parent.parent.parent / "examples" / "demo_vendor_pack.zip"
            if not demo_pack_path.exists():
                QMessageBox.warning(self, "Error", "Static demo vendor pack not found.")
                return

            spec_l = VendorPackSpec(
                vendor="POSM-DEMO",
                adapter="coilcraft_csv",
                source_path=demo_pack_path,
                glob_pattern="**/demo_inductors.csv",
                measurement_plane="EOM_external_RF_connector"
            )
            spec_c = VendorPackSpec(
                vendor="POSM-DEMO",
                adapter="murata_csv",
                source_path=demo_pack_path,
                glob_pattern="**/demo_capacitors.csv",
                measurement_plane="EOM_external_RF_connector"
            )
            # Dummy s2p imports to attach measured model tier
            spec_l_s2p = VendorPackSpec(
                vendor="POSM-DEMO",
                adapter="s2p_coilcraft",
                source_path=demo_pack_path,
                glob_pattern="**/*-L-*.s2p",
                measurement_plane="EOM_external_RF_connector"
            )
            spec_c_s2p = VendorPackSpec(
                vendor="POSM-DEMO",
                adapter="s2p_murata_gjm_gqm",
                source_path=demo_pack_path,
                glob_pattern="**/*-C-*.s2p",
                measurement_plane="EOM_external_RF_connector"
            )

            self._lib_path = path
            try:
                manifest_l = LibraryCtrl.import_pack(spec_l, self._lib_path)
                manifest_c = LibraryCtrl.import_pack(spec_c, self._lib_path)
                LibraryCtrl.import_pack(spec_l_s2p, self._lib_path)
                LibraryCtrl.import_pack(spec_c_s2p, self._lib_path)
                self._load(self._lib_path)

                # Show structured report for the demo import
                report = "Demo Library Created!\n\n"
                report += f"Inductors Imported: {manifest_l.n_inserted_total}\n"
                report += f"Capacitors Imported: {manifest_c.n_inserted_total}\n"
                QMessageBox.information(self, "Demo Library", report)
            except Exception as e:
                QMessageBox.warning(self, "Import Error", str(e))


    def _import_pack(self) -> None:
        if not self._lib_path:
            return

        dlg = ImportVendorPackDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if not dlg.source_path or not dlg.vendor_input.text():
                QMessageBox.warning(self, "Invalid Input", "Vendor name and source path are required.")
                return

            spec = VendorPackSpec(
                vendor=dlg.vendor_input.text().strip(),
                adapter=dlg.adapter_combo.currentText(),
                source_path=Path(dlg.source_path),
                glob_pattern=dlg.glob_input.text().strip() or "**/*.*"
            )
            try:
                manifest = LibraryCtrl.import_pack(spec, self._lib_path)
                self._load(self._lib_path)
                self._show_import_report(manifest)
            except Exception as e:
                QMessageBox.warning(self, "Import Error", f"Import failed without corrupting the DB:\\n{e}")

    def _show_import_report(self, manifest) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Structured Import Report")
        lay = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)
        report = []
        report.append(f"Total discovered: {manifest.n_files_processed}")
        report.append(f"Imported: {manifest.n_inserted_total}")
        report.append(f"Skipped/deduplicated: {manifest.n_skipped_dup_total}")
        report.append(f"Errors: {manifest.n_error_total}")
        report.append("\nWarnings/Errors:")
        if manifest.all_errors:
            for err in manifest.all_errors:
                report.append(f"- {err}")
        else:
            report.append("None")

        text.setPlainText("\n".join(report))
        lay.addWidget(text)

        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()

    def _apply_filters(self) -> None:
        if not self._lib_path:
            return

        q = ComponentQuery()
        txt = self.search_input.text().strip()
        if txt:
            q.part_number_glob = f"%{txt}%"

        vnd = self.vendor_combo.currentText()
        if vnd != "All":
            q.vendor = vnd

        knd = self.kind_combo.currentText()
        if knd == "Inductor":
            q.kind = ComponentKind.INDUCTOR
        elif knd == "Capacitor":
            q.kind = ComponentKind.CAPACITOR

        vmin = self.val_min_input.text().strip()
        vmax = self.val_max_input.text().strip()
        if vmin:
            with contextlib.suppress(ValueError):
                q.value_min = float(vmin)
        if vmax:
            with contextlib.suppress(ValueError):
                q.value_max = float(vmax)

        tier = self.tier_combo.currentText()
        if tier != "All":
            q.model_tier_min = ModelTier(tier)

        fmin = self.freq_min_input.text().strip()
        fmax = self.freq_max_input.text().strip()
        if fmin and fmax:
            with contextlib.suppress(ValueError):
                q.freq_range_hz = (float(fmin), float(fmax))

        self._load(self._lib_path, query=q)

    def _clear_filters(self) -> None:
        self.search_input.clear()
        self.vendor_combo.setCurrentIndex(0)
        self.kind_combo.setCurrentIndex(0)
        self.val_min_input.clear()
        self.val_max_input.clear()
        self.freq_min_input.clear()
        self.freq_max_input.clear()
        self.tier_combo.setCurrentIndex(0)
        if self._lib_path:
            self._load(self._lib_path)

    def _load(self, path: str, query: ComponentQuery | None = None) -> None:
        try:
            self._stats = LibraryCtrl.get_stats(path, query)
            self._lib_path = path
            self._populate()
            self.btn_import.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _populate(self) -> None:
        if not self._stats:
            return
        s = self._stats
        self.lbl_path.setText(f"Path: {self._lib_path}")
        self.lbl_counts.setText(
            f"Total Parts: {s.total_parts} | Inductors: {s.n_inductors} | Capacitors: {s.n_capacitors} | "
            f"Measured: {s.n_measured} | Parametric: {s.n_parametric} | Ideal: {s.n_ideal}"
        )
        self.lbl_sha.setText(f"Library SHA-256: {s.sha256}")

        self.table.setRowCount(len(s.parts))
        for i, p in enumerate(s.parts):
            self.table.setItem(i, 0, QTableWidgetItem(p.vendor))
            self.table.setItem(i, 1, QTableWidgetItem(p.part_number))
            self.table.setItem(i, 2, QTableWidgetItem(p.kind))
            self.table.setItem(i, 3, QTableWidgetItem(p.value))
            self.table.setItem(i, 4, QTableWidgetItem(p.tier))
            self.table.setItem(i, 5, QTableWidgetItem(p.validity))
            self.table.setItem(i, 6, QTableWidgetItem(p.origin))
            # Store ID in user data of the first column
            self.table.item(i, 0).setData(Qt.ItemDataRole.UserRole, p.id)

    def _on_selection_changed(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        if not self.table.selectionModel().hasSelection():
            self._clear_details()
            return

        row = self.table.selectionModel().selectedRows()[0].row()
        item = self.table.item(row, 0)
        if not item:
            return

        comp_id = item.data(Qt.ItemDataRole.UserRole)
        if not comp_id or not self._lib_path:
            return

        try:
            vm = LibraryCtrl.get_component_details(self._lib_path, comp_id)
            self.lbl_det_vendor.setText(vm.vendor)
            self.lbl_det_part.setText(vm.part_number)
            self.lbl_det_kind.setText(vm.kind)
            self.lbl_det_pkg.setText(vm.package)
            self.lbl_det_val.setText(vm.value_nom)
            self.lbl_det_tol.setText(vm.tolerance)
            self.lbl_det_esr.setText(vm.dcr_esr)
            self.lbl_det_srf.setText(vm.srf)
            self.lbl_det_vrating.setText(vm.voltage_rating)
            self.lbl_det_irating.setText(vm.current_rating)
            self.lbl_det_validity.setText(vm.validity_hz)
            self.lbl_det_tier.setText(vm.model_tier)
            self.lbl_det_origin.setText(vm.model_origin)
            self.lbl_det_sha.setText(vm.model_file_sha256)

            if vm.is_synthetic:
                self.lbl_det_warning.setText("SYNTHETIC DEMO COMPONENT DATA — NOT FOR HARDWARE DESIGN")
            else:
                self.lbl_det_warning.setText("")
        except Exception as e:
            self._clear_details()
            print(f"Error loading details: {e}")

    def _clear_details(self) -> None:
        self.lbl_det_vendor.setText("N/A")
        self.lbl_det_part.setText("N/A")
        self.lbl_det_kind.setText("N/A")
        self.lbl_det_pkg.setText("N/A")
        self.lbl_det_val.setText("N/A")
        self.lbl_det_tol.setText("N/A")
        self.lbl_det_esr.setText("N/A")
        self.lbl_det_srf.setText("N/A")
        self.lbl_det_vrating.setText("N/A")
        self.lbl_det_irating.setText("N/A")
        self.lbl_det_validity.setText("N/A")
        self.lbl_det_tier.setText("N/A")
        self.lbl_det_origin.setText("N/A")
        self.lbl_det_sha.setText("N/A")
        self.lbl_det_warning.setText("")

    # ------------------------------------------------------------------
    @property
    def library_path(self) -> str | None:
        return self._lib_path

    @property
    def library_sha(self) -> str | None:
        return self._stats.sha256 if self._stats else None

    def set_library(self, path: str | None) -> None:
        """Programmatically set the library (e.g. on project load)."""
        if path:
            self._load(path)
