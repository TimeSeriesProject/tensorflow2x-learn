

import tensorflow as tf

# ************* 常用函数 tf.nn.softmax  ****************
# 将一个向量tensor, 转化为概率输出

a = tf.constant([1, 3, 6], dtype=tf.float32)
prob = tf.nn.softmax(a)
print(prob)