
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

print(test_x[0])
print('y', test_y[0])


model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=L2()))
model.add(Dense(128, activation='relu', kernel_regularizer=L2()))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y), validation_freq=4)

model.summary()
