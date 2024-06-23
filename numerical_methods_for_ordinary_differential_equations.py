
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from matplotlib.font_manager import FontProperties

# 常微分方程数值求解方法
## 基于微分方程的数值求解方法
### 前向欧拉法
def forward_euler(f, y0: float, x0: float, x_end: float, h:float):
    """
    前向欧拉法求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(t0) = y0
        t0: 初始条件
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        # 前向欧拉法公式
        y[i] = y[i-1] + h * f(x[i-1], y[i-1])

    return x, y


exp = np.exp
cos = np.cos
sin = np.sin
log = np.log
power = np.power

# 后退欧拉法
from scipy.optimize import fsolve
def backward_euler(f, y0, x0, x_end, h):
    """
    后退欧拉法求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(x0) = y0
        x0: 初始点x0
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        # 定义隐式方程
        yi = lambda yi: yi - y[i-1] - h * f(x[i], yi)
        # 求解 yi)= 0
        y[i] = fsolve(yi, y[i-1])

    return x, y

### 预测-校正方法
def euler_predict_correct(f, y0, x0, x_end, h):
    """
    预测-校正方法求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(x0) = y0
        x0: 初始点
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        # 使用前向欧拉法进行预测
        y_predict = y[i-1] + h * f(x[i-1], y[i-1])
        # 最后使用后退欧拉法进行校正
        y[i] = y[i-1] + h* f(x[i], y_predict)

    return x, y

## 基于积分的数值求解方法
### 梯形法
def trapezoidal_method(f, y0, x0, x_end, h):
    """
    梯形法求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(x0) = y0
        x0: 初始点
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        # 梯形法公式
        yi=lambda yi:yi-y[i-1]-0.5*h*(f(x[i-1],y[i-1])+f(x[i],yi))
        y[i]=fsolve(yi,y[i-1])
    return x, y

# 梯形法:预测-校正方法
def trapezoidal_predict_correct(f, y0, x0, x_end, h):
    """
    梯形法:预测-校正方法求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(x0) = y0
        x0: 初始点
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y_predict=np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        # 使用前向欧拉法进行预测
        y_predict[i] = y[i-1] + h * f(x[i-1], y[i-1])
        # 最后使用梯形法进行校正
        y[i] = y[i-1] +0.5*h* (f(x[i-1], y[i-1]) + f(x[i], y_predict[i]))

    return x, y

# Simpson公式
def simpson_method(f, y0, x0, x_end, h):
    """
    Simpson公式求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(x0) = y0,y(x1)=y1
        x0: 初始点,初始点为[x0,x1]
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    #Simpson是两步公式，因此需要两个初始值
    assert len(x0)==2
    assert len(y0)==2
    h1=h/2
    x = np.arange(x0[0], x_end + h, h1)
    y = np.zeros(len(x))
    y[:2] = y0
    for i in range(2, len(x)):
        # Simpson公式
        g=lambda yi:yi-y[i-2]-h1/3*(f(x[i-2],y[i-2])+4*f(x[i-1],y[i-1])+f(x[i],yi))
        y[i] = fsolve(g, y[i-1])
    return x, y

# 四阶Runge-Kutta法
def runge_Kutta(f, y0, x0, x_end, h):
    """
    四阶Runge-Kutta法求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y0: 初始条件 y(x0) = y0
        x0: 初始点
        x_end: 终止点
        h: 步长
    Return:
        x: 自变量数组
        y: 数值解数组
    """
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        K1 = f(x[i-1], y[i-1])
        K2 = f(x[i-1] + h*0.5, y[i-1] + h*0.5 * K1)
        K3 = f(x[i-1] + h*0.5, y[i-1] + h*0.5 * K2)
        K4 = f(x[i-1] + h, y[i-1] + h * K3)
        y[i] = y[i-1] + h/6 * (K1 + 2*K2 + 2*K3 + K4)
    return x, y

## 线性多步法
### Adams显式求解公式
def adams_explicit(f,y:np.ndarray,x:np.ndarray,x_end:int,h: float):
    """
    Adams显式求解公式求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y: 数值解数组
        x: 自变量数组
        h: 步长
    Return:
        y: 数值解数组
    """
    # Adams显式求解公式需要前三个点
    assert len(y)==3
    assert len(x)==3
    x_steps=np.arange(x[0],x_end+h,h)
    steps=len(x_steps)
    y_adams_explicit=np.zeros(steps)
    y_adams_explicit[0:3]=y
    for i in range(3, len(x_steps)):
        # Adams显式求解公式
        y_adams_explicit[i] = y_adams_explicit[i-1]+h/12 * \
            (23*f(x_steps[i-1], y_adams_explicit[i-1])-16*f(x_steps[i-2], y_adams_explicit[i-2])+5*f(x_steps[i-3], y_adams_explicit[i-3]))
    return x_steps,y_adams_explicit

# Adams隐式求解公式
def adams_implicit(f, y: np.ndarray, x: np.ndarray, x_end: int, h: float):
    """
    Adams显式求解公式求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y: 数值解数组
        x: 自变量数组
        h: 步长
    Return:
        y: 数值解数组
    """
    x_steps = np.arange(x[0], x_end+h, h)
    steps = len(x_steps)
    y_adams_implicit = np.zeros(steps)
    y_adams_implicit[0:3] = y
    for i in range(3, len(x_steps)):
        # Adams隐式求解公式
        yi=lambda yi:yi-y_adams_implicit[i-1]-h/24*(9*f(x_steps[i],yi)+19*f(x_steps[i-1],y_adams_implicit[i-1])-5*f(x_steps[i-2],y_adams_implicit[i-2])+f(x_steps[i-3],y_adams_implicit[i-3]))
        # 求解yi=0
        y_adams_implicit[i]=fsolve(yi,y_adams_implicit[i-1])
    return x_steps, y_adams_implicit

# Adams 预测矫正系统
def adams_predict_correct(f, y0: np.ndarray, x0: np.ndarray, x_end: int, h: float):
    """
    Adams显式求解公式求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y: 数值解数组
        x: 自变量数组
        h: 步长
    Return:
        y: 数值解数组
    """
    x = np.arange(x0[0], x_end+h, h)
    steps = len(x)
    y = np.zeros(steps)
    y[0:3] = y0
    for i in range(3, len(x)):
        #预测
        y_predict= y[i-1]+h/24*(55*f(x[i-1],y[i-1])-59*f(x[i-2],y[i-2])+37*f(x[i-3],y[i-3])-9*f(x[i-4],y[i-4]))
        #矫正
        y[i] = y[i-1]+h/24*(9*f(x[i],y_predict)+19*f(x[i-1],y[i-1])-5*f(x[i-2],y[i-2])+f(x[i-3],y[i-3]))
    return x, y


def milne_method(f, y0: np.ndarray, x0: np.ndarray, x_end: int, h: float):
    """
    Milne求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y: 数值解数组
        x: 自变量数组
        h: 步长
    Return:
        y: 数值解数组
    """
    x = np.arange(x0[0], x_end+h, h)
    steps = len(x)
    y = np.zeros(steps)
    y[0:4] = y0
    for i in range(4, len(x)):
        y[i] = y[i-4]+4*h/3*(2*f(x[i-1],y[i-1])-f(x[i-2],y[i-2])+2*f(x[i-3],y[i-3]))
    return x, y

def milne_predict_correct(f, y: np.ndarray, x: np.ndarray, x_end: int, h: float):
    """
    Milne预测矫正系统求解常微分方程
    Params:
        f: 函数 f(x, y)，表示微分方程 y' = f(x, y)
        y: 数值解数组
        x: 自变量数组
        h: 步长
    Return:
        y: 数值解数组
    """
    x_steps = np.arange(x[0], x_end+h, h)
    steps = len(x_steps)
    y_milne_predict_correct = np.zeros(steps)
    y_milne_predict_correct[0:4] = y
    for i in range(4, len(x_steps)):
        y_predict = y_milne_predict_correct[i-4]+4*h/3*(2*f(x_steps[i-1],y_milne_predict_correct[i-1])-f(x_steps[i-2],y_milne_predict_correct[i-2])+2*f(x_steps[i-3],y_milne_predict_correct[i-3]))
        y_milne_predict_correct[i] = y_milne_predict_correct[i-2]+h/3*(f(x_steps[i],y_predict)+4*f(x_steps[i-1],y_milne_predict_correct[i-1])+f(x_steps[i-2],y_milne_predict_correct[i-2]))
    return x_steps, y_milne_predict_correct
