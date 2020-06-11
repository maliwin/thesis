import tensorflow as tf
from util import *
preload_tensorflow()

from art.attacks.evasion import FastGradientMethod
from art.utils import to_categorical
from art.classifiers import TensorFlowV2Classifier
from art.defences.preprocessor import ThermometerEncoding


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range

num_space = 5
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3 * num_space), activation='relu', padding='same'),
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
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# @tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

defence = ThermometerEncoding(clip_values=(0, 1), num_space=num_space)
art_model = TensorFlowV2Classifier(probability_model, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                   preprocessing_defences=defence, train_step=train_step,
                                   loss_object=loss_object)

art_model.fit(x_train[:300], to_categorical(y_train[:300], 10), nb_epochs=1)

attack = FastGradientMethod(art_model)
attack.generate(defence(x_test[:1]))
