
import tensorflow as tf
from sklearn import datasets
import numpy as np
import time
import matplotlib.pyplot as plt

iris_data = datasets.load_iris()
data_x = iris_data.data
data_y = iris_data.target

np.random.seed(116)
idxs = np.arange(data_x.shape[0])
print("sample_num: {}".format(idxs.shape[0]))
np.random.shuffle(idxs)
tf.random.set_seed(116)

train_x = tf.cast(data_x[idxs[:-30]], tf.float32)
train_y = tf.cast(data_y[idxs[:-30]], tf.int32)
#
test_x = tf.cast(data_x[idxs[-30:]], tf.float32)
test_y = tf.cast(data_y[idxs[-30:]], tf.int32)

print('test_x.shape: ', test_y.shape[0])
print('test_y', test_y)
train_x = tf.cast(train_x, tf.float32)
test_x = tf.cast(test_x, tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)

print(test_y)
lr = 0.1
epochs = 500
train_loss = []
test_acc = []

w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

start_time = time.time()
for epoch in range(epochs):

    epoch_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(batch_x, w) + b
            y = tf.nn.softmax(y)

            # one-hot编码
            true_y = tf.one_hot(batch_y, depth=3)
            loss = tf.reduce_mean(tf.square(y, true_y))
            epoch_loss += float(loss)

        grad = tape.gradient(loss, [w, b])
        w.assign_sub(lr*grad[0])
        b.assign_sub(lr*grad[1])
        # print('step: %d' % step)

    print('train epoch: %d, loss: %.3f' % (epoch, epoch_loss/4))
    train_loss.append(epoch_loss/4)

    sample_num, correct_num = 0, 0
    for step, (batch_x, batch_y) in enumerate(test_db):
        y = tf.matmul(batch_x, w) + b
        y = tf.nn.softmax(y)

        y = tf.argmax(y, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(y, tf.int32), batch_y), tf.int32))
        # print('correct ', correct)
        correct_num += int(correct)
        sample_num += batch_x.shape[0]

    acc = correct_num/sample_num
    test_acc.append(acc)
    print('test sample_num: %d, correct_num: %d, acc: %.3f' % (sample_num, correct_num, acc))
    print('----------------\n')

total_time = time.time() - start_time
print('total_time: %.3f' % total_time)

plt.title('loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_loss, label='$loss$')
plt.legend()
plt.show()

plt.title('acc curve')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.plot(test_acc, label='$acc$')
plt.legend()
plt.show()
