

import tensorflow as tf
import numpy as np

# ************* 常用函数  ****************
# ************ 1. 强制tensor转换为该数据类型
# tf.cast(张量名, dtype=数据类型)
a = tf.constant([1, 2, 3], dtype=tf.float64)
print(a)
a2 = tf.cast(a, tf.int32)
print(a2)


# **********  tf.Variable ****************
# tf.Variable()将变量标记为"可训练", 被标记的变量会在方向传播中记录梯度信息,神经网络训练中,常用该函数标记待训练参数
# tf.Variable(初始值)
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
print('w', w)
print(w.numpy())

# ************** tf.one_hot ******************
# tf.one_hot()函数将数据转换为ont-hot形式的数据: tf.one_hot(待转换数据,depth=分类数量)
print(" ************ one-hot *********************")
label = tf.constant([1, 0, 2])  # 输入的最小值为0, 最大值为2
result = tf.one_hot(label, depth=3)
print(result)


# *************** assign_sub ***************
# 估值操作, 更新参数的值并返回
# 调用assign_sub前, 先用tf.Variable定义变量为可训练(自更新)
# w.assign_sub(w要自减的内容)
print('*********  assign_sub*****************')
w = tf.Variable([[2, 3],
                 [4, 5]])
w.assign_sub([[1, 2],
              [1, 2]])
print(w)


# *************** tf.argmax ***************
# 返回张量沿指定维度最大值的索引(下表)
# tf.argmax(张量名, axis=要计算的维度)
print('*********  argmax *****************')

x = np.array([[7, 2, 3],
              [5, 8, 6],
              [7, 10, 9]])
print(x)
print(tf.argmax(x, axis=0))  # 返回纵向的最大索引
print(tf.argmax(x, axis=1))  # 返回很想的最大索引

