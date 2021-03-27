
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

print(test_x[0])
print('y', test_y[0])


# 自定义model类
class FashionModel(Model):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu', kernel_regularizer=L2())
        self.d2 = Dense(128, activation='relu', kernel_regularizer=L2())
        self.d3 = Dense(10, activation='softmax')

    def call(self, x):
        y = self.flatten(x)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
        return y


model = FashionModel()

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y), validation_freq=4)

model.summary()
