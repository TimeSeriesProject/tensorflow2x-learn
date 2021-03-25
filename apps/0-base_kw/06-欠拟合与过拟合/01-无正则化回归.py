
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = '../../../datasets/dot.csv'
pd_data = pd.read_csv(data_file)

print(pd_data)
data_x = np.array(pd_data[['x1', 'x2']])
data_y = np.array(pd_data[['y_c']])

print(data_x.shape)
print(data_y.shape)

# data_x = np.vstack(data_x).reshape(-1, 2)
# data_y = np.vstack(data_y).reshape(-1, 1)

train_x = tf.cast(data_x, tf.float32)
train_y = tf.cast(data_y, tf.float32)

# 数据打包
train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)

# 训练参数设置
lr = 0.005
epochs = 10


# 神经网络构建
w1 = tf.Variable(tf.random.truncated_normal([2, 8], mean=0, stddev=0.1), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[8]))

w2 = tf.Variable(tf.random.truncated_normal([8, 1], mean=0, stddev=0.1), dtype=tf.float32)
b2 = tf.Variable(tf.constant([0.01]), dtype=tf.float32)

for epoch in range(0, epochs):
    for idx, (batch_x, batch_y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            hidden = tf.matmul(batch_x, w1) + b1
            hidden = tf.nn.relu(hidden)
            y = tf.matmul(hidden, w2) + b2

            loss = tf.reduce_mean(tf.square(batch_y - y))

        grad = tape.gradient(loss, [w1, b1, w2, b2])
        w1.assign_sub(lr * grad[0])
        b1.assign_sub(lr * grad[1])
        w2.assign_sub(lr * grad[2])
        b2.assign_sub(lr * grad[3])

    if epoch % 10 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss))


# 见训练数据当成预测数据进行预测
sample_num = 0
loss = 0
for idx, (test_x, test_y) in enumerate(train_db):
    y = tf.matmul(test_x, w1) + b1
    print(y)
    y = tf.nn.relu(y)
    y = tf.matmul(y, w2) + b2

    loss += tf.reduce_sum(tf.square(test_y, y))
    sample_num += test_x.shape[0]

print("loss: %f, sample_num: %d, mse: %f" % (loss, sample_num, loss/sample_num))


# 重新生成数据预测
# xx, yy在-3到3之间以步长为0.01， yy在-3到3之间以步长0.01， 生成间隔数值点
xx, yy = np.mgrid[-3:3:0.1, -3:3:0.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点

grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

probs = []
for test_x in grid:
    y = tf.matmul([test_x], w1) + b1
    y = tf.nn.relu(y)
    y = tf.matmul(y, w2) + b2
    probs.append(y)

# 拆分成两个坐标
x1 = data_x[:, 0]
x2 = data_x[:, 1]

# probs的shape调整
Y_c = [['red' if y else 'blue'] for y in data_y]
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))

# 将坐标xx, yy和对应的值probs放入contour函数， 给probs值为0.5的所有点上色
plt.contour(xx, yy, probs, levels=[0.5])
plt.show()
