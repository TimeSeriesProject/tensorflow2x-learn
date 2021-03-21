
import numpy as np

# ********* np.mgrid[] ************
# np.mgrid[起始值:结束值:步长, 起始值:结束值:步长, ...]  [起始值,结束值)左闭右开

# x.ravel() 将x变成一维数组

# np.c_[] 是返回的间隔数值点配对, 笛卡尔积方式连接
# np.c_[数组1, 数组2, 数组3, ....]

x, y, z = np.mgrid[1:3:1, 2:4:0.5, 3:5:1]
print('x\n', x)
print('y\n', y)
print('z\n', z)
print('x.ravel\n', x.ravel())

grid = np.c_[x.ravel(), y.ravel()]
print('grid\n', grid)
