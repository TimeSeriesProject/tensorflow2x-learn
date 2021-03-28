
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


(train_x, train_y), (test_x, test_y) = mnist.load_data()

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


checkpoint_save_path = './checkpoint/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------------- load model ----------------')
    model.load_weights(checkpoint_save_path)

model_save_callback = ModelCheckpoint(
    checkpoint_save_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch')

history = model.fit(train_x, train_y, batch_size=32, epochs=20,
                    validation_data=(test_x, test_y), validation_freq=1,
                    callbacks=[model_save_callback])

model.summary()

# acc, loss提取
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# 画第一个图
plt.subplot(1, 2, 1)
plt.plot(acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.title('accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.title('loss')
plt.legend()

plt.show()
