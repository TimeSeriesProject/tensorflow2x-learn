
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from tensorflow.keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x/255.0, test_x/255.0
print('train data:', train_x.shape, train_y.shape)
print('test data: ', test_x.shape, test_y.shape)

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y), validation_freq=5)

model.summary()
