import glob
import ntpath
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


EPOCH_COUNT = 15
FORCE_RETRAIN = False


def untrained_model_pytorch():
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1))
            self.conv2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
            self.conv3 = nn.Conv2d(32, 64, 3, padding=(1, 1))
            self.conv4 = nn.Conv2d(64, 64, 3, padding=(1, 1))
            self.dropout25 = nn.Dropout(p=0.25)
            self.dropout50 = nn.Dropout(p=0.50)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.dropout25(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = self.dropout25(x)
            x = x.view(-1, 64 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.dropout50(x)
            x = self.fc2(x)
            return x

    net = Net()
    return net


def get_untrained_model_tf(input_shape=(32, 32, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(10)
    ])
    return model


def make_model_tf(epochs=EPOCH_COUNT, preload_attempt=True, save_on_preload_fail=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range

    def _train_new_model():
        model = get_untrained_model_tf()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        return model

    modelname = 'tf_cifar10_epochs%d' % epochs
    path = './saved_models/' + modelname
    if FORCE_RETRAIN:
        model = _train_new_model()
        model.save(path)
    else:
        if preload_attempt:
            try:
                model = tf.keras.models.load_model(path)
            except:
                model = _train_new_model()
                if save_on_preload_fail:
                    model.save(path)
        else:
            model = _train_new_model()
            model.save(path)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    return model, probability_model, (x_train, y_train), (x_test, y_test)


def cifar10_class_id_to_text(class_ids):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    names = []
    for class_id in class_ids:
        names.append(classes[class_id])
    return names


def display_images(images, grid_shape, figsize=None, titles=None):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=grid_shape, axes_pad=0.3)
    if not titles:
        titles = [None, ] * len(images)

    for ax, im, title in zip(grid, images, titles):
        ax.imshow(im)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title:
            ax.title.set_text(title)
    plt.show()


def resize_image(image, new_size):
    scale = new_size / max(image.size)
    scaled_image = Image.new(image.mode, (new_size, new_size), (0, 0, 0))
    resized = image.resize((int(image.width * scale), int(image.height * scale)), resample=Image.LANCZOS)
    scaled_image.paste(resized, (0, 0))
    return scaled_image


def load_images(directory_path, size=None, crop=False):
    filelist = glob.glob(directory_path + '/*.jpg')
    images = []
    for f in filelist:
        image = Image.open(f)
        if crop:
            scaled_image = ImageOps.fit(image, size, Image.ANTIALIAS, bleed=0.1)
        else:
            # scaled_image = resize_image(image, size[0])
            scaled_image = image.resize(size, resample=Image.LANCZOS)
        images.append(np.array(scaled_image))
    return np.array(images)


def preprocessing_art_tuple(mode, shape=(224, 224, 3)):
    assert mode in ('tf', 'torch', 'caffe')

    if mode == 'tf':
        return 127.5, 127.5

    if mode == 'torch':
        preprocessing_mean = np.zeros(shape)
        preprocessing_mean[..., 0].fill(123.675)
        preprocessing_mean[..., 1].fill(116.28)
        preprocessing_mean[..., 2].fill(103.53)

        preprocessing_std = np.zeros(shape)
        preprocessing_std[..., 0].fill(0.229 * 255)
        preprocessing_std[..., 1].fill(0.224 * 255)
        preprocessing_std[..., 2].fill(0.225 * 255)

        return preprocessing_mean, preprocessing_std

    if mode == 'caffe':
        # NB: caffe is BGR, not RGB
        preprocessing_mean = np.zeros(shape)
        preprocessing_mean[..., 0].fill(103.939)
        preprocessing_mean[..., 1].fill(116.779)
        preprocessing_mean[..., 2].fill(123.68)

        return preprocessing_mean, 1


def save_numpy_arrays(np_array, name, directory='results'):
    path = os.path.join(directory, name)
    np.save(path, np_array)


def load_numpy_array(name, directory='results'):
    path = os.path.join(directory, name + '.npy')
    return np.load(path)


def preload_tensorflow():
    # nobody should ever ask me what this does. ever.
    # x1 = tf.random.normal((1,))
    # x2 = tf.random.normal((1,))
    # y = tf.keras.layers.Conv2D()([x1, x2])
    import tensorflow as tf
    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)
    y = tf.keras.layers.Conv2D(
        2, 3, activation='relu', input_shape=input_shape)(x)
    print('loaded tf')


if __name__ == '__main__':
    save_numpy_arrays(np.array([1, 2, 3]), 'test')
    t = load_numpy_array('test')
