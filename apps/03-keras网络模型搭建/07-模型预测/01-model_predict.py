
import os
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import L2
from PIL import Image
import numpy as np

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_x, test_x = train_x/255.0, test_x/255.0
print('train_x.shape:', train_x.shape)


model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=L2()))
model.add(Dense(10, activation='softmax'))


checkpoint_save_path = './checkpoint/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------------- load model ----------------')
    model.load_weights(checkpoint_save_path)
else:
    print('----------------- not found model weight, will use init weight predict ----------------')


print(' ********************************* predict ***************************************')
predict_data_path = '../../../../datasets/tensorflow2x-learn/mnist_image_label/predict_imgs'


if not os.path.exists(predict_data_path):
    print("data path not exist")
    exit(0)

for root, dirs, files in os.walk(predict_data_path):
    for img_file in files:
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            # read img
            img = Image.open(os.path.join(root, img_file))
            # resize
            img = img.resize((28, 28), Image.ANTIALIAS)
            img_arr = np.array(img.convert('L'))

            # 变成黑底白色字体，保持与模型训练样本一致
            img_data = [[255 if img_arr[i][j] < 200 else 0 for j in range(28)] for i in range(28)]
            img_arr = np.array(img_data)
            img_arr = img_arr / 255.0

            img_arr = img_arr[tf.newaxis, ...]
            result = model.predict(img_arr)
            predict_num = tf.argmax(result, axis=1)
            print('%s predict result' % img_file, predict_num.numpy()[0])
