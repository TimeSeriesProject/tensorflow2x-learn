

import tensorflow as tf

# **********  tensorflow中的数学运算 ****************
# ****** 对应元素的四则运算, 只有维度相同的张量才可以做四则运算
# tf.add(张量1, 张量2),
# tf.substract(张量1, 张量2),
# tf.multiply,
# tf.devide

a = tf.ones([2, 3], dtype=tf.float32)
b = tf.fill([2, 3], 5)
b = tf.cast(b, dtype=tf.float32)
print(a)
print(b)
print(tf.add(a, b))
print(tf.subtract(a, b))
print(tf.multiply(a, b))
print(tf.divide(a, b))

# ******* 平方,次方,开放:tf.square, tf.pow, tf.sqrt
a = tf.fill([2, 3], 2.0)
print(a)
print(tf.pow(a, 3))
print(tf.square(a))
print(tf.sqrt(a))


# ******** 矩阵乘: tf.matmul
a = tf.ones([2, 3])
b = tf.fill([3, 4], 3.)
print(tf.matmul(a, b))
