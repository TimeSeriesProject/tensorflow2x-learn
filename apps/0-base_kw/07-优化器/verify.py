
import tensorflow as tf
import numpy as np

# 数字与tensor的结合计算
m_w = 0.5
beta = 0.5

grad = tf.cast([1, 2, 3], tf.float32)
print('m_w', grad)
m_w_new = beta * m_w + (1 - beta) * grad
print('m_w_new', m_w_new)


# one-hot 下标与index的关系
print(' ---------------- one-hot ------------------')
data = np.array([2, 0, 1, 2, 0])
data = data.reshape((-1, ))

print('data.shape', data.shape)

"""
depth=2， 直接label中大于等于depth直接过滤掉，使用代替， 生成结果:
[[1. 0.]
 [0. 0.]
 [0. 1.]
 [0. 0.]
 [1. 0.]]
"""
result = tf.one_hot(data, depth=2)
print('result', result)

"""
depth=3， 与预期结果相符:
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
"""
result = tf.one_hot(data, depth=3)
print('result', result)

"""
depth=4， 补位操作:
[[1. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]]
"""
result = tf.one_hot(data, depth=4)
print('result', result)

