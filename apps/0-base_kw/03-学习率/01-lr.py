

import tensorflow as tf
epochs = 20
LR_BASE = 0.2
LR_DECAY = 0.99
LR_STEP = 1

# y = x^2
w = tf.Variable(tf.random.truncated_normal([1], mean=0, stddev=5, dtype=tf.float32))
for epoch in range(epochs):
    #  根据迭代次数调整学习率
    lr = LR_BASE * LR_DECAY**(epoch/LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)
    w.assign_sub(lr*grads)
    print("after %s epoch, w is %f, loss is %f, lr is %f" % (epoch, w.numpy(), loss.numpy(), lr))
    # print("after % epoch, w is %f, loss is %f, lr is %f" % (epoch, w.numpy, loss, lr))
