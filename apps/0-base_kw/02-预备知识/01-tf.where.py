

import tensorflow as tf
# ****** tf.where() ***********
# tf.where(条件语句, 真返回A, 假返回B), 条件语句真返回A,条件语句假返回B, A和B的shape必须相同

a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])

# 依次比较a, b对应的元素, ,若a>b, 则返回a中的元素,  否则返回b中的元素,形成新的tensor
c = tf.where(tf.greater(a, b), a, b)
print(c)
print(c.numpy())


# 多为shape比较
a = tf.constant([[1, 2],
                [1, 1]])
b = tf.constant([[0, 1],
                 [4, 5]])

# 依次比较a, b对应的元素, ,若a>b, 则返回a中的元素,  否则返回b中的元素,形成新的tensor
c = tf.where(tf.greater(a, b), a, b)
print(c)
print(c.numpy())