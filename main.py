# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets
from main_ui import Ui_BasicForm
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

from numerical_methods_for_ordinary_differential_equations import *


def create_function(window):
    expression = window.fun.toPlainText().strip()
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
        QtWidgets.QMessageBox.critical(window, "错误", f"生成函数时出错: {e}")


def apply_clicked(window):
    algorithm = window.algorithms.currentText()
    x0 = window.input_2.text()
    x_end = window.input_3.text()
    h = window.input_4.text()
    y0 = window.input.text()
    f = window.create_function()
    if x0 == "" or x_end == "" or h == "" or y0 == "":
        window.please_input.setText("请输入全部条件")
        window.please_input.show()
    else:
        x0 = int(x0)
        x_end = int(x_end)
        h = float(h)
        if len(y0) > 1:
            y0 = [int(i) for i in y0.split()]
        else:
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
        window.please_input.setText("请选择算法")
        window.please_input.show()
        return
    print(f"{x=},{y[-1]=}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_BasicForm()
    ui.setupUi(MainWindow)
    MainWindow.show()
    MainWindow.setWindowTitle("常微分方程数值解")
    sys.exit(app.exec())
