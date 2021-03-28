
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


dataset_root = '../../../../datasets/tensorflow2x-learn'


def load_data(dataset_name='mnist', dataset_root=None):
    dataset_root = '../../../../datasets/tensorflow2x-learn' if dataset_root is None else dataset_root
    # result = _generate_dataset_path(dataset_name, dataset_root)
    # print(result)
    train_path, train_txt, train_x_save_path, train_y_save_path, test_path, test_txt, test_x_save_path, test_y_save_path = _generate_dataset_path(dataset_name, dataset_root)
    print(train_x_save_path)
    if os.path.exists(train_x_save_path) and os.path.exists(train_y_save_path) and \
            os.path.exists(test_x_save_path) and os.path.exists(test_y_save_path):
        print("-------------- file exist, loading dataset ----------------")
        train_x = np.load(train_x_save_path)
        train_y = np.load(train_y_save_path)
        test_x = np.load(test_x_save_path)
        test_y = np.load(test_y_save_path)

        train_x = np.reshape(train_x, (train_x.shape[0], 28, 28))
        test_x = np.reshape(test_x, (test_x.shape[0], 28, 28))
    else:
        print('--------------- generate dataset ----------------')
        train_x, train_y = _generate_npy(train_path, train_txt)
        test_x, test_y = _generate_npy(test_path, test_txt)

        print('------------ save dataset -----------------')
        train_x_save = np.reshape(train_x, (train_x.shape[0], -1))
        test_x_save = np.reshape(test_x, (test_x.shape[0], -1))
        np.save(train_x_save_path, train_x_save)
        np.save(train_y_save_path, train_y)
        np.save(test_x_save_path, test_x_save)
        np.save(test_y_save_path, test_y)
    return (train_x, train_y), (test_x, test_y)


def _generate_dataset_path(dataset_name, dataset_root):
    train_path = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_train_jpg_60000' % dataset_name)
    train_txt = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_train_jpg_60000.txt' % dataset_name)
    train_x_save_path = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_train_x.npy' % dataset_name)
    train_y_save_path = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_train_y.npy' % dataset_name)

    test_path = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_test_jpg_10000' % dataset_name)
    test_txt = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_test_jpg_10000.txt' % dataset_name)
    test_x_save_path = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_test_x.npy' % dataset_name)
    test_y_save_path = os.path.join(dataset_root, '%s_image_label' % dataset_name, '%s_test_y.npy' % dataset_name)
    return train_path, train_txt, train_x_save_path, train_y_save_path, test_path, \
           test_txt, test_x_save_path, test_y_save_path


def _generate_npy(img_path, label_txt):
    with open(label_txt, 'r') as f:
        contents = f.readlines()
    x, y = [], []
    for idx in tqdm(range(len(contents))):
        content = contents[idx]
        value = content.split()
        img_file = os.path.join(img_path, value[0])
        img = Image.open(img_file)
        img = np.array(img.convert('L'))

        x.append(img)
        y.append(value[1])

    x = np.array(x)
    y = np.array(y)
    y = y.astype(np.int64)
    return x, y


# (x, y), (tx, ty) = load_data('mnist')
(x, y), (tx, ty) = load_data('fashion')
