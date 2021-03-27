
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import L2

from data_load import load_data

# load data, can set dataset_name for 'mnist' or 'fashion'
# (train_x, train_y), (test_x, test_y) = load_data(dataset_name='mnist')
(train_x, train_y), (test_x, test_y) = load_data(dataset_name='fashion')
train_x, test_x = train_x/255.0, test_x/255.0

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=L2()))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y), validation_freq=4)

model.summary()
