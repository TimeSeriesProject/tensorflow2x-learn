

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import L2

import numpy as np

# 设置不省略的打印
np.set_printoptions(threshold=np.inf)

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_x, test_x = train_x/255.0, test_x/255.0
print('train_x.shape:', train_x.shape)


model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=L2()))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
model.fit(train_x, train_y, batch_size=32, epochs=10,
          validation_data=(test_x, test_y), validation_freq=4)


print('model.variables\n', model.trainable_variables)

with open('./model_weight.txt', 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

model.summary()
