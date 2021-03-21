
import numpy as np

# ************* np.random.RandomState.rand() ***********
# 返回一个[0, 1]之间的随机数
# np.random.RandomState.rand(维度)  # 维度为空, 返回标量

# 设置随机种子,可以保证每次运行的结果一直
rdm = np.random.RandomState(seed=1)

a = rdm.rand()   # 返回一个随机标量
print(a)

b = rdm.rand(2, 3)  # 返回一个维度为2行3列的随机矩阵
print(b)
