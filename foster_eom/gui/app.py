"""Main GUI Application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from foster_eom.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Foster-Schmidt Opt")

    window = MainWindow()
    window.resize(1024, 768)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
