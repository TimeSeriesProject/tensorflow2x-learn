
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import L2

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_x, test_x = train_x/255.0, test_x/255.0
print('train_x.shape:', train_x.shape)

train_image_gen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1.0,
)

# ****************  需要转为思维 *******************
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
train_image_gen.fit(train_x)

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=L2()))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
model.fit(train_image_gen.flow(train_x, train_y, batch_size=32), epochs=20,
          validation_data=(test_x, test_y), validation_freq=4)


model.summary()
