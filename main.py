# -*- coding: utf-8 -*-
from functools import partial
from PyQt6 import QtWidgets
from main_ui import Ui_BasicForm
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from main_ui import Ui_BasicForm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numerical_methods_for_ordinary_differential_equations import *
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


def plot_to_frame(ui: Ui_BasicForm):
    if hasattr(ui, 'sc'):
        ui.frame_2.layout().removeWidget(ui.sc)
        ui.sc.setParent(None)
    ui.sc = MplCanvas(ui.frame_2, width=9, height=3.3, dpi=100)
    ui.sc.axes.plot(ui.x, ui.y)
    if ui.frame_2.layout() is None:
        layout = QtWidgets.QVBoxLayout(ui.frame_2)
        layout.addWidget(ui.sc)
    else:
        ui.frame_2.layout().addWidget(ui.sc)
    ui.sc.draw()


def quitAPP():
    sys.exit()


def emptize(ui: Ui_BasicForm):
    ui.input.clear()
    ui.input_2.clear()
    ui.input_3.clear()
    ui.input_4.clear()
    ui.fun.clear()
    # ui.plainTextEdit_2.clear()
    ui.result.clear()

    # Clear all widgets in frame_2
    for i in reversed(range(ui.frame_2.layout().count())):
        widget_to_remove = ui.frame_2.layout().itemAt(i).widget()
        if widget_to_remove is not None:
            widget_to_remove.setParent(None)


def create_function(ui: Ui_BasicForm):
    expression = ui.fun.toPlainText().strip()
    try:
        # 定义函数字符串
        function_str = f"def f(x,y): return {expression}"
        # 在局部命名空间中执行函数字符串定义函数
        local_vars = {
            "exp": exp,
            "power": power,
            "cos": cos,
            "sin": sin,
            "log": log,
        }
        exec(
            function_str,
            {
                "np": np,
                "exp": exp,
                "power": power,
                "cos": cos,
                "sin": sin,
                "log": log,
            },
            local_vars,
        )
        # 从局部命名空间中获取函数对象
        func = local_vars["f"]
        return func
    except Exception as e:
        QtWidgets.QMessageBox.critical(ui, "错误", f"生成函数时出错: {e}")


def apply_clicked(ui:Ui_BasicForm):
    algorithm = ui.algorithms.currentText()
    x0 = ui.input_2.text()
    x_end = ui.input_3.text()
    h = ui.input_4.text()
    y0 = ui.input.text()
    f = create_function(ui)
    if x0 == "" or x_end == "" or h == "" or y0 == "":
        ui.please_input.setText("请输入全部条件")
        ui.please_input.show()
        return None
    else:
        x_end = int(x_end)
        h = float(h)
        if len(y0) > 1:
            y0 = [float(i) for i in y0.split()]
            x0 = [float(i) for i in x0.split()]
        else:
            x0 = int(x0)
            y0 = int(y0)
    if algorithm == "前向欧拉法":
        x, y = forward_euler(f, y0, x0, x_end, h)
    elif algorithm == "后退欧拉法":
        x, y = backward_euler(f, y0, x0, x_end, h)
    elif algorithm == '欧拉法:预测-校正系统':
        x, y = euler_predict_correct(f, y0, x0, x_end, h)
    elif algorithm == '梯形法':
        x, y = trapezoidal_method(f, y0, x0, x_end, h)
    elif algorithm == '梯形法:预测-校正系统':
        x, y = trapezoidal_predict_correct(f, y0, x0, x_end, h)
    elif algorithm == 'Simpson法':
        x, y = simpson_method(f, y0, x0, x_end, h)
    elif algorithm == '四阶Runge-Kutta法':
        x, y = runge_Kutta(f, y0, x0, x_end, h)
    elif algorithm == 'Adams显式求解公式':
        x, y = adams_explicit(f, y0, x0, x_end, h)
    elif algorithm == 'Adams隐式求解公式':
        x, y = adams_implicit(f, y0, x0, x_end, h)
    elif algorithm == 'Adams法:预测矫正系统':
        x, y = adams_predict_correct(f, y0, x0, x_end, h)
    elif algorithm == 'Milne法':
        x, y = milne_method(f, y0, x0, x_end, h)
    elif algorithm == 'Milne法:预测-矫正系统':
        x, y = milne_predict_correct(f, y0, x0, x_end, h)
    else:
        ui.please_input.setText("请选择算法")
        ui.please_input.show()
        return
    ui.x = x
    ui.y = y
    print(f"{len(ui.x)=},{len(ui.y)=}")
    plot_to_frame(ui)

    # 显示结果
    ui.result.setRowCount(2)
    ui.result.setColumnCount(len(x))
    ui.result.setHorizontalHeaderLabels(["x", "y"])

    for i in range(len(x)):
        ui.result.setItem(0, i, QTableWidgetItem(str(x[i])))
        ui.result.setItem(1, i, QTableWidgetItem(str(y[i])))
    ui.retranslateUi(ui.BasicForm)



def bound(ui: Ui_BasicForm):
    # 传递ui参数给apply_clicked
    ui.apply.clicked.connect(partial(apply_clicked, ui))
    ui.empty.clicked.connect(partial(emptize, ui))
    ui.quit.clicked.connect(quitAPP)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_BasicForm()
    ui.setupUi(MainWindow)
    bound(ui)
    MainWindow.show()
    MainWindow.setWindowTitle("常微分方程数值解")
    sys.exit(app.exec())
