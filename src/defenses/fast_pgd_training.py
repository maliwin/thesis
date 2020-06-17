import torch
import keras
import numpy as np

from keras.utils import to_categorical
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from small_resnet.small_resnet_pytorch import resnet_18
from art.classifiers import PyTorchClassifier
from art.defences.trainer import AdversarialTrainerFBFPyTorch

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def fast_pgd():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32).swapaxes(1, 3)
    y_test = y_test.flatten()
    x_test = x_test.astype(np.float32).swapaxes(1, 3)

    model = resnet_18()
    art_model = PyTorchClassifier(model, loss=CrossEntropyLoss(), optimizer=SGD(model.parameters(),
                                                                                lr=0.21, momentum=0.9, weight_decay=5e-4),
                                  input_shape=(3, 32, 32), nb_classes=10, device_type='gpu', clip_values=(0, 255))
    trainer = AdversarialTrainerFBFPyTorch(art_model, eps=8/255)
    trainer.fit(x_train, to_categorical(y_train, 10), nb_epochs=20,
                validation_data=(x_test[:1000], to_categorical(y_test[:1000], 10)))
    torch.save(model.state_dict(), 'saved_models/pytorch_fbf_cifar10_small_resnet_epochs20')
    b = art_model.predict(x_train)
    a = 5


def train_regular():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32).swapaxes(1, 3)
    y_test = y_test.flatten()
    x_test = x_test.astype(np.float32).swapaxes(1, 3)

    model = resnet_18()
    art_model = PyTorchClassifier(model, loss=CrossEntropyLoss(), optimizer=SGD(model.parameters(),
                                                                                lr=0.21, momentum=0.9, weight_decay=5e-4),
                                  input_shape=(3, 32, 32), nb_classes=10, device_type='gpu')
    art_model.fit(x_train, to_categorical(y_train, 10), nb_epochs=20)
    torch.save(model.state_dict(), 'saved_models/pytorch_cifar10_small_resnet_epochs20')
    print(np.sum(np.argmax(art_model.predict(x_test), axis=1) == y_test.flatten()) / len(y_test.flatten()))


def evaluate_fast_pgd_training():
    from art.attacks.evasion import FastGradientMethod
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32).swapaxes(1, 3)
    x_test = x_test.astype(np.float32).swapaxes(1, 3)
    y_test = y_test.flatten()

    model1 = resnet_18()
    model1.load_state_dict(torch.load('saved_models/pytorch_cifar10_small_resnet_epochs15'))
    model1.eval()

    model2 = resnet_18()
    model2.load_state_dict(torch.load('saved_models/pytorch_fbf_cifar10_small_resnet_epochs15'), strict=False)
    model2.eval()


    art_model1 = PyTorchClassifier(model1, loss=CrossEntropyLoss(), optimizer=SGD(model1.parameters(),
                                                                                lr=0.21, momentum=0.9,
                                                                                weight_decay=5e-4),
                                  input_shape=(3, 32, 32), nb_classes=10, device_type='gpu')
    print(np.sum(np.argmax(art_model1.predict(x_test), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    y_pred1 = np.argmax(art_model1.predict(x_test), axis=1)
    i1 = np.where(y_pred1 == y_test)[0]

    art_model2 = PyTorchClassifier(model2, loss=CrossEntropyLoss(), optimizer=SGD(model2.parameters(),
                                                                                lr=0.21, momentum=0.9,
                                                                                weight_decay=5e-4),
                                  input_shape=(3, 32, 32), nb_classes=10, device_type='gpu')
    print(np.sum(np.argmax(art_model2.predict(x_test), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    y_pred2 = np.argmax(art_model2.predict(x_test), axis=1)
    i2 = np.where(y_pred2 == y_test)[0]

    a1 = FastGradientMethod(art_model1, eps=5)
    a2 = FastGradientMethod(art_model2, eps=5)
    a1.generate(x_test[i1])
    a2.generate(x_test[i2])


if __name__ == '__main__':
    evaluate_fast_pgd_training()
    # train_regular()
