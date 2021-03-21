
import tensorflow as tf
from sklearn import datasets
import numpy as np
from apps.common.plot_data.plot import plot_loss


iris_data = datasets.load_iris()
data_x = iris_data.data
data_y = iris_data.target

# ******* 打乱数据方法1 **********
np.random.seed(116)
rnd_idx = np.arange(len(data_x))
np.random.shuffle(rnd_idx)

train_x = data_x[rnd_idx[:-30]]
train_y = data_y[rnd_idx[:-30]]
valid_x = data_x[rnd_idx[-30:]]
valid_y = data_y[rnd_idx[-30:]]

# ******** 打算数据方法2 **************
# np.random.seed(116)
# np.random.shuffle(data_x)
# np.random.seed(116)
# np.random.shuffle(data_y)
# tf.random.set_seed(116)
#
# train_x = data_x[:-30]
# train_y = data_y[:-30]
# valid_x = data_x[-30:]
# valid_y = data_y[-30:]


train_db = tf.data.Dataset.from_tensor_slices((tf.constant(train_x, dtype=tf.float32),
                                               tf.constant(train_y, dtype=tf.int32))).batch(32)
valid_db = tf.data.Dataset.from_tensor_slices((tf.constant(valid_x, dtype=tf.float32),
                                               tf.constant(valid_y, dtype=tf.int32))).batch(32)


lr = 0.1   # 值太小容易收敛慢,设置为1e-3,需要上万次训练才能收敛.
epochs = 500
train_loss = []
valid_acc = []

# 定义变量
w = tf.Variable(tf.random.truncated_normal([4, 3], mean=0, stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.random.truncated_normal([3], mean=0, stddev=0.1, dtype=tf.float32))

for epoch in range(0, epochs):
    batch_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            pre_y = tf.matmul(batch_x, w) + b
            pre_y = tf.nn.softmax(pre_y)
            y_ = tf.one_hot(batch_y, depth=3)
            loss = tf.reduce_mean(tf.square(tf.subtract(y_, pre_y)))
        grad = tape.gradient(loss, [w, b])
        w.assign_sub(lr * grad[0])
        b.assign_sub(lr * grad[1])
        batch_loss += loss
    avg_loss = batch_loss/4
    print('epoch: {}, train_loss: {}'.format(epoch, avg_loss))
    train_loss.append(avg_loss)

    correct = 0
    total_sample = 0
    for valid_x, valid_y in valid_db:
        pre_y = tf.add(tf.matmul(valid_x, w), b)
        pre_y = tf.nn.softmax(pre_y)
        pre_label = tf.cast(tf.argmax(pre_y, axis=1), valid_y.dtype)

        correct += int(tf.reduce_sum(tf.cast(tf.equal(pre_label, valid_y), dtype=tf.int32)))
        total_sample += valid_x.shape[0]
    print('acc: {}'.format(correct/total_sample))
    valid_acc.append(correct / total_sample)

plot_loss({'loss': train_loss}, key_list=['loss'])
plot_loss({'acc': valid_acc}, key_list=['acc'])


print(train_loss)
print(valid_acc)
