from __future__ import absolute_import, division, print_function

import os
import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from eit_app import resource_rc
from backend import UiBackEnd


def main():
    """Run the eit_app"""
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # TODO test icon dipslay on win
    app.setWindowIcon(QtGui.QIcon(":/icons/icons/EIT.png"))
    ui = UiBackEnd()
    ui.show()
    # exit(app.exec_())
    sys.exit(app.exec_())


if __name__ == "__main__":
    from glob_utils.log.log import main_log

    main_log()
    main()
