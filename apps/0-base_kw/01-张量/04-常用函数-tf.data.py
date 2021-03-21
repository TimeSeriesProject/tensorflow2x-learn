

import tensorflow as tf
import numpy as np

# ************* 常用函数 tf.data.Dataset.from_tensor_slices  ****************
# 切分传入张量的第一维度,生成输入特征/标签对,构建数据集
# data = tf.data.Dataset.from_tensor_slices((输入特征,标签)), numpy和tensor格式都可用该语句读入数据

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6],
                 [7, 8],
                 [9, 10]], dtype=tf.float32)
y = tf.constant([0, 0, 1, 1, 0])

dataset = tf.data.Dataset.from_tensor_slices((x, y))

for ele in dataset:
    print(ele)
