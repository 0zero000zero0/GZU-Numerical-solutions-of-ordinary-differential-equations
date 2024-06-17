# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets
from main_ui import Ui_BasicForm
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

from numerical_methods_for_ordinary_differential_equations import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_BasicForm()
    ui.setupUi(MainWindow)
    MainWindow.show()
    MainWindow.setWindowTitle("常微分方程数值解")
    sys.exit(app.exec())
