
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

iris_data = datasets.load_iris()

data_x = iris_data.data
data_y = iris_data.target

np.random.seed(116)
np.random.shuffle(data_x)
np.random.seed(116)
np.random.shuffle(data_y)
tf.random.set_seed(1)


# 自定义模型
class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2())

    def call(self, x):
        y = self.d1(x)
        return y


model = IrisModel()

lr = 0.1
model.compile(optimizer=SGD(lr=0.1),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

batch_size = 32
epochs = 500
split_rate = 0.2
model.fit(data_x, data_y, batch_size=32, epochs=epochs, validation_split=split_rate, validation_freq=10)

model.summary()

