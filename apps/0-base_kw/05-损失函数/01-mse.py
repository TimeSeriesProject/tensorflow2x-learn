
# *********** 损失函数(loss) ************
"""
预测值与已知答案的察隅
NN优化目标: loss最小

mse
交叉熵损失函数CE(cross entropy)
自定义损失函数
"""

import tensorflow as tf
import numpy as np

SEED = 1
COST = 1
PROFIT = 1

rnd = np.random.RandomState(seed=SEED)
x = rnd.rand(32, 2)
y = [x1 + x2 + (rnd.rand()/10.0 - 0.05) for (x1, x2) in x]

x = tf.cast(x, tf.float32)

lr = 0.002
epochs = 15000

w = tf.Variable(tf.random.truncated_normal([2, 1], dtype=tf.float32, mean=0, stddev=1, seed=1))
# 加上偏置的模型
b = tf.Variable(tf.constant(0, dtype=tf.float32))


for epoch in range(epochs):
    with tf.GradientTape() as tape:
        _y = tf.matmul(x, w) + b
        loss = tf.reduce_mean(tf.square(y - _y))
    # grads = tape.gradient(loss, [w, b])
    grads = tape.gradient(loss, [w])
    w.assign_sub(lr*grads[0])
    # b.assign_sub(lr*grads[1])
    if epoch % 1000:
        print('epoch: %s, w: %s, b: %s, loss: %s' % (epoch, w.numpy(), b.numpy(), loss.numpy()))
print("train over")


