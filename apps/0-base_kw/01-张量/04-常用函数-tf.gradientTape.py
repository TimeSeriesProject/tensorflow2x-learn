

import tensorflow as tf

# ************* 常用函数 tf.GradientTape  ****************
# gradient计算张量的梯度
# with tf.GradientTape() as tape:
#     计算
# grap = tape.gradient(函数, 对谁求导)

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant([2, 3], dtype=tf.float32))
    y = tf.pow(w, 3)
grad = tape.gradient(y, w)
print(grad)  # [12, 37]
