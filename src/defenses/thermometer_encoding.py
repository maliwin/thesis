import sys
import time

from util import *
preload_tensorflow()
import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical

from art.attacks.evasion import DeepFool
from art.defences.preprocessor import ThermometerEncoding
from attacks.fgm import fgm


def thermo():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    modelname = 'thermometer_cifar10_30_postfix_10space'
    path = '../saved_models/' + modelname
    model = tf.keras.models.load_model(path)

    defence = ThermometerEncoding(clip_values=(0, 1), num_space=10)
    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 30), clip_values=(0, 1),
                                        preprocessing_defences=defence, train_step=train_step,
                                        loss_object=loss_object)

    z = np.argmax(art_model.predict(x_test), axis=1) == y_test.flatten()
    a, b = fgm(art_model, x_test[:10], eps=0.03)

    model2, _, _, _ = setup_cifar10_model(epochs=25)
    art_model2 = TensorFlowV2Classifier(model2, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                        train_step=train_step, loss_object=loss_object)
    a2, b2 = fgm(art_model2, x_test[:10], eps=0.03)


if __name__ == '__main__':
    setup_logging()
    thermo()
