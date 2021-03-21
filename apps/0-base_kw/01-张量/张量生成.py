

import tensorflow as tf
import numpy as np


# ************  1.创建一个tensor ******************
# tf.constant(张量内容，dtype=数据类型(可选))， 如下：
a = tf.constant([1, 5], dtype=tf.int32)
print(a)
print(a.dtype)
print(a.shape)
print(a.get_shape())

# 输入tensor的值
print(a.numpy())


# ************  2. convert_to_tensor ******************
# ************  将numpy的数据类型转换为tensor数据类型 ******************
# tf.convert_to_tensor(数据名, dtype=数据类型(可选))
print('*************** 2. convert_to_tensor ****************')

a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int32)
print(a)
print(b)
print(b.numpy())


# ************  2. tf.zeros/tf.ones/tf.fill 创建张量 ******************
# 创建全为0的张量：tf.zeros(维度)
# 创建全为1的张量：tf.ones(维度)
# 创建全为指定值的张量: tf.fill(维度，指定值)
print("********  tf.zeros/tf.ones/tf.fill 创建张量 **************")
a = tf.zeros([2, 3])
print(a)
print(a.numpy())

b = tf.ones(4)
print(b)
print(b.numpy())

c = tf.fill([2, 2], 9)
print(c)
print(c.numpy())


# ************  4. 生成随机值张量  ******************
# ********** 生成正态分布的随机数，默认均值为0， 标准差为1
# tf.random.normal(维度, mean=均值, stddev=标准差)
print("************ 生成随机值的张量 ******************")
a = tf.random.normal([2, 3], mean=0, stddev=1, dtype=tf.float16)
print(a)
print(a.numpy())

# (*************** 生成截断式正态分布的随机数
# tf.random.truncated_normal(维度, mean=均值, stddev=标准差)
# truncated_normal中如果随机生成的数据取值在(mean-2*sigsma, mean+2*sigma)之外,则重新生成,保证了生成值在均值附近
b = tf.random.truncated_normal([2, 3], mean=0.5, stddev=1, dtype=tf.float32)
print(b)
print(b.numpy())

# (*************** 生成均匀分布的随机数
# tf.random.uniform(维度, minval=最小值, maxval=最大值)
c = tf.random.uniform([2, 3], minval=2, maxval=5, dtype=tf.float32)
print(c)
print(c.numpy())
print(c.numpy)
