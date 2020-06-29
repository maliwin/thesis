import sys
import time

from util import *
preload_tensorflow()
# setup_logging()

import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer


def simple_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    already_trained_model, _, _, _ = setup_cifar10_model(20)
    model, _ = get_untrained_model_tf()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    art_model = TensorFlowV2Classifier(model=model, loss_object=model.loss,
                                       nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                       train_step=train_step)

    attack = FastGradientMethod(art_model, eps=0.1)
    adv_trainer = AdversarialTrainer(art_model, attacks=[attack])
    adv_trainer.fit(x_train, to_categorical(y_train, 10), nb_epochs=20)

    a = 5


def test_fgsm(model):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test, y_train = y_test.flatten(), y_train.flatten()
    x_train, x_test = x_train / 255, x_test / 255

    art_model = TensorFlowV2Classifier(model=model, loss_object=model.loss,
                                       nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1))

    y_pred = np.argmax(model.predict(x_test), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('acc %.4f' % acc)

    attacks = FastGradientMethod(art_model, eps=2 / 255)
    adv = attacks.generate(x_test)
    y_pred = np.argmax(model.predict(adv), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('acc %.4f' % acc)

    attacks = FastGradientMethod(art_model, eps=5 / 255)
    adv = attacks.generate(x_test)
    y_pred = np.argmax(model.predict(adv), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('acc %.4f' % acc)

    attacks = FastGradientMethod(art_model, eps=10 / 255)
    adv = attacks.generate(x_test)
    y_pred = np.argmax(model.predict(adv), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('acc %.4f' % acc)

    print('-'*30)


def adv_train(eps=0.1, epochs=20, name='unnamed'):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test, y_train = y_test.flatten(), y_train.flatten()
    x_train, x_test = x_train / 255, x_test / 255

    model, _ = get_untrained_model_tf()
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer,
                  loss=loss_object,
                  metrics=['accuracy'])

    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    art_model = TensorFlowV2Classifier(model=model, loss_object=model.loss,
                                       nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                       train_step=train_step)

    try:
        attacks = [FastGradientMethod(art_model, eps=e) for e in eps]
    except:
        attacks = [FastGradientMethod(art_model, eps=eps)]

    adv_trainer = AdversarialTrainer(art_model, attacks=attacks)
    adv_trainer.fit(x_train, to_categorical(y_train, 10), nb_epochs=epochs)
    print('- - FGSM trained model - -')
    acc = np.sum(np.argmax(model.predict(x_test), axis=1) == y_test) / len(y_test)
    print('Clean accuracy: %.5f' % acc)
    model.save('fgsm_training/' + name)


if __name__ == '__main__':
    # setup_logging()
    # adv_train(epochs=25, eps=2/255, name='epochs25_01_eps_2')
    # adv_train(epochs=50, eps=2/255, name='epochs50_01_eps_2')
    # adv_train(epochs=25, eps=5/255, name='epochs25_01_eps_5')
    # adv_train(epochs=50, eps=5/255, name='epochs50_01_eps_5')
    # adv_train(epochs=25, eps=10/255, name='epochs25_01_eps_10')
    # adv_train(epochs=50, eps=10/255, name='epochs50_01_eps_10')
    # adv_train(epochs=25, eps=[2/255, 5/255, 10/255], name='epochs25_01_eps_all')
    # adv_train(epochs=50, eps=[2/255, 5/255, 10/255], name='epochs50_01_eps_all')
    # model = tf.keras.models.load_model('fgsm_training/' + 'epochs25_01_eps_2')
    # test_fgsm(model)
    # model = tf.keras.models.load_model('fgsm_training/' + 'epochs50_01_eps_2')
    # test_fgsm(model)
    # model = tf.keras.models.load_model('fgsm_training/' + 'epochs25_01_eps_5')
    # test_fgsm(model)
    # model = tf.keras.models.load_model('fgsm_training/' + 'epochs50_01_eps_5')
    # test_fgsm(model)
    # model = tf.keras.models.load_model('fgsm_training/' + 'epochs25_01_eps_10')
    # test_fgsm(model)
    # model = tf.keras.models.load_model('fgsm_training/' + 'epochs50_01_eps_10')
    # test_fgsm(model)
    model = tf.keras.models.load_model('fgsm_training/' + 'epochs25_01_eps_all')
    test_fgsm(model)
    model = tf.keras.models.load_model('fgsm_training/' + 'epochs50_01_eps_all')
    test_fgsm(model)
