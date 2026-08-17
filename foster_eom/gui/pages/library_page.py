"""Library management page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from foster_eom.gui.controllers.library_ctrl import LibraryCtrl
from foster_eom.gui.view_models.library_vm import LibraryStats


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
        self.btn_import = QPushButton("Import Vendor Pack…")
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self._import_pack)
        tb.addWidget(self.btn_open)
        tb.addWidget(self.btn_create)
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
        lay.addWidget(self.table, 1)

    # ------------------------------------------------------------------
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

    def _import_pack(self) -> None:
        if not self._lib_path:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vendor Pack",
            "",
            "ZIP archives (*.zip);;Directories (*);;All (*)",
        )
        if path:
            try:
                LibraryCtrl.import_pack(path, self._lib_path)
                self._load(self._lib_path)
                QMessageBox.information(self, "Import", "Vendor pack imported successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Import Error", str(e))

    def _load(self, path: str) -> None:
        try:
            self._stats = LibraryCtrl.get_stats(path)
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
            f"Total: {s.total_parts}  |  "
            f"Measured: {s.n_measured}  |  "
            f"Parametric: {s.n_parametric}  |  "
            f"Ideal: {s.n_ideal}"
        )
        self.lbl_sha.setText(f"SHA-256: {s.sha256}")

        self.table.setRowCount(len(s.parts))
        for i, p in enumerate(s.parts):
            self.table.setItem(i, 0, QTableWidgetItem(p.vendor))
            self.table.setItem(i, 1, QTableWidgetItem(p.part_number))
            self.table.setItem(i, 2, QTableWidgetItem(p.kind))
            self.table.setItem(i, 3, QTableWidgetItem(p.value))
            self.table.setItem(i, 4, QTableWidgetItem(p.tier))
            self.table.setItem(i, 5, QTableWidgetItem(p.validity))
            self.table.setItem(i, 6, QTableWidgetItem(p.origin))

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
