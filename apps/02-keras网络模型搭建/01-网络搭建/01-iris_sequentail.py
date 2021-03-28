
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
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

model = Sequential()
model.add(Dense(3, input_shape=(None, 4), dtype=float, activation='softmax'))

lr = 0.1
model.compile(optimizer=SGD(lr=0.1),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

batch_size = 32
epochs = 500
split_rate = 0.2
model.fit(data_x, data_y, batch_size=32, epochs=epochs, validation_split=split_rate, validation_freq=10)

model.summary()

