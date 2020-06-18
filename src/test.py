from util import *
preload_tensorflow()

import tensorflow as tf
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == '__main__':
    val = datasets.ImageFolder(
        '../data/imagenet_validation',
        transforms.Compose([
            transforms.Resize(256, Image.BICUBIC),
            transforms.CenterCrop(224),
        ]))

    model = tf.keras.applications.resnet_v2.ResNet50V2()
    y = np.array(val.targets)
    x = np.array([np.array(img[0]) for img in val])
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    y_pred = np.argmax(model.predict(x), axis=1)
    print(np.sum(y == y_pred) / len(y))

    val = datasets.ImageFolder(
        '../data/imagenet_validation',
        transforms.Compose([
            transforms.Resize(331, Image.BICUBIC),
            transforms.CenterCrop(299),
        ]))

    model = tf.keras.applications.Xception()
    y = np.array(val.targets)
    x = np.array([np.array(img[0]) for img in val])
    x = tf.keras.applications.xception.preprocess_input(x)
    y_pred = np.argmax(model.predict(x), axis=1)
    print(np.sum(y == y_pred) / len(y))

    a = 5
