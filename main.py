# -*- coding: utf-8 -*-
from main_ui import Ui_BasicForm
import  sys
from PyQt6.QtWidgets import QApplication, QMainWindow
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_BasicForm()
    ui.setupUi(MainWindow)
    MainWindow.show()
    MainWindow.setWindowTitle("常微分方程数值解")
    sys.exit(app.exec())