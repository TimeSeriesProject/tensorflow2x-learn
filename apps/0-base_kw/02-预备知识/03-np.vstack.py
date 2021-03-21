
# *********** np.vsstack  ***********
# 将两个数组按垂直方向叠加
# np.vstack(数组1, 数组2)

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print(c)


# 多个数组垂直方向叠加
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])
d = np.vstack((a, b, c))
print(d)

# 错误示范, 数组维度必须保持一致
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6, 7])
# c = np.vstack((a, b))
# print(c)
