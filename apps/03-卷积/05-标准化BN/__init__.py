

import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten, Conv2D, BatchNormalization, Activation

model = tf.keras.models.Sequential([
  Conv2D(filters=6, kernel_size=(5, 5), padding='same'),
  BatchNormalization(),  # 添加BN层
  Activation('relu'),
  MaxPool2D(pool_size=(2, 2), strides=2),
  Flatten(),
  Dense(10, activation='sigmoid')
])
