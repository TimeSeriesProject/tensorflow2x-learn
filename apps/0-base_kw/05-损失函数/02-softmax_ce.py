

# 交叉熵损失函数CE(cross entropy)


import tensorflow as tf

res = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
print(res)
res = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print(res)


# softmax与交叉熵结合
# tf.nn.softmax_cross_entropy_with_logits(真正标签,预测结果)

y = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]]

y_pre = [[12, 3, 2],
         [3, 10, 1],
         [1, 2, 5],
         [4, 6.5, 1.2],
         [3, 6, 1]]

# 分开计算
y_prob = tf.nn.softmax(y_pre)
print(y_prob)
res = tf.losses.categorical_crossentropy(y, y_prob)
print('分开计算结果\n', res)

# 直接调用函数
res = tf.nn.softmax_cross_entropy_with_logits(y, y_pre)
print('直接计算结果\n', res)
