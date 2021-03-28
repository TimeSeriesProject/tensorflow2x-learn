
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

train_x, test_x = train_x/255.0, test_x/255.0
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=[SparseCategoricalAccuracy()])


# save model
model_save_path = './checkpoint/baseline.ckpt'
model_save_callback = ModelCheckpoint(
    model_save_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch'
)
# load model
if os.path.exists(model_save_path + '.index'):
    print('*************** load model *********************************')
    model.load_weights(model_save_path)


history = model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y),
                    validation_freq=1,
                    callbacks=[model_save_callback])

model.summary()

# ************************ 保存权重 *************
mode_weight_txt = './weight.txt'
with open(mode_weight_txt, 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

# ************ plot loass/acc curve ****************************
loss, val_loss = history.history['loss'], history.history['val_loss']
acc, val_acc = history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy']

plt.subplot(1, 2, 1)
plt.plot(loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.title('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.title('accuracy')
plt.legend()

plt.show()
