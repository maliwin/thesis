import sys
import time

from util import *
preload_tensorflow()
setup_logging()

import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

model, _ = get_untrained_model_tf()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

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


art_model = TensorFlowV2Classifier(model=probability_model, loss_object=model.loss,
                                   nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                   train_step=train_step)

already_trained_model = setup_cifar10_model(20)
attack = FastGradientMethod(art_model, eps=0.05)
adv_trainer = AdversarialTrainer(art_model, attacks=[attack])
adv_trainer.fit(x_train, to_categorical(y_train, 10), nb_epochs=20)

adversarial_images = attack.generate(x_test[:9])
correct_predictions = y_test[:9]
nonadversarial_predictions = np.argmax(art_model.predict(x_test[:9]), axis=1)
adversarial_predictions = np.argmax(art_model.predict(adversarial_images), axis=1)

x_train_adv = x_train[:25000]
x_train_adv = attack.generate(x_train_adv)
new_x_train = np.concatenate((x_train_adv, x_train[25000:]))

model, _ = get_untrained_model_tf()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

a = 5
