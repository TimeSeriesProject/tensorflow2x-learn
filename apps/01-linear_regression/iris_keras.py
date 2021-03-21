
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MSE
import numpy as np

from sklearn import datasets

from apps.common.plot_data.plot import plot_loss
from apps.common.utils import calcu_acc

iris_data = datasets.load_iris()

# print iris_data info
# print(iris_data.data)
# print(len(iris_data.target))
# print(iris_data.target_names)
# print(iris_data)

# one-hot encode
label = to_categorical(iris_data.target, num_classes=3)
# print(label)

# shuffle data
np.random.seed(1)
shuffle_idx = np.arange(len(label))
np.random.shuffle(shuffle_idx)

# split data for train and valid
train_len = int(len(shuffle_idx) * 0.8)
train_x = iris_data.data[shuffle_idx[:train_len]]
train_y = label[shuffle_idx[:train_len]]

valid_x = iris_data.data[shuffle_idx[train_len:]]
valid_y = label[shuffle_idx[train_len:]]

# build net work
model = Sequential()

# network model
# model.add(Dense(16, input_shape=(4, )))
# model.add(Dense(8))
# model.add(Dense(3, activation='sigmoid'))

# linear regression
model.add(Dense(3, input_shape=(4,), activation='sigmoid'))

# print model structure
model.summary()


model.compile(optimizer='adam', loss=MSE, metrics=['accuracy'])
history = model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), epochs=3000, verbose=True)

# plot history
plot_loss(history.history, key_list=['loss', 'val_loss', 'accuracy', 'val_accuracy'],
          need_save=True, save_path='./xx.png')

predict_result = model.predict(valid_x)
print(predict_result)
predict = np.argmax(predict_result, axis=1)
t_label = np.argmax(valid_y, axis=1)
print(len(t_label), len(predict))

print("acc:", calcu_acc(predict, t_label))
print(predict)
