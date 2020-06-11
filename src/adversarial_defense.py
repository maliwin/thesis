import sys
import time

from util import *
preload_tensorflow()
import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical

setup_logging()

def fgsm_adv_training():
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
    a = 5
    # TODO: get labels

    x_train_adv = x_train[:25000]
    x_train_adv = attack.generate(x_train_adv)
    new_x_train = np.concatenate((x_train_adv, x_train[25000:]))

    model, _ = get_untrained_model_tf()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    a = 5


def jpeg_compression():
    from art.attacks.evasion import FastGradientMethod
    from art.defences.preprocessor import JpegCompression

    model, art_model, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()
    attack = FastGradientMethod(art_model, eps=5)
    no_jpeg = attack.generate(images)

    defense = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=50)
    model, art_model, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(preprocessing_defences=[defense])
    attack = FastGradientMethod(art_model, eps=5)
    jpeg = attack.generate(images)
    a = 5


def thermometer():
    from art.utils import to_categorical
    from art.classifiers import TensorFlowV2Classifier
    from art.defences.preprocessor import ThermometerEncoding

    model, probability_model = get_untrained_model_tf(input_shape=(32, 32, 15))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')

    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(train_loss(loss))
        # train_accuracy(labels, predictions)

    defence = ThermometerEncoding(clip_values=(0, 1), num_space=5)
    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 15), clip_values=(0, 1),
                                       preprocessing_defences=defence, train_step=train_step,
                                       loss_object=model.loss)

    # z, _ = defence(x_train[:100])
    import time
    t1 = time.time()
    art_model.fit(x_train[:300], to_categorical(y_train[:300], 10), nb_epochs=1)
    t2 = time.time()
    print('time %f' % (t2 - t1))

    from art.attacks.evasion import ProjectedGradientDescent
    attack = ProjectedGradientDescent(art_model, eps=20)
    attack.generate(x_test[:2])
    a = 5


def adversarial_pgd():
    from art.utils import to_categorical
    from art.classifiers import TensorFlowV2Classifier
    from art.defences.trainer import AdversarialTrainerMadryPGD

    # model, probability_model = setup_cifar10_model()
    model, probability_model, _, _ = setup_cifar10_model()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)
    # x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 255),
                                       loss_object=loss_object, train_step=train_step, preprocessing=(0, 255))
    trainer = AdversarialTrainerMadryPGD(art_model, nb_epochs=30, eps=8, eps_step=2, num_random_init=1)
    import time
    t1 = time.time()
    trainer.fit(x_train, y_train)
    t2 = time.time()
    print('time %f' % (t2 - t1))
    a = 5


def squeeze():
    from art.attacks.evasion import FastGradientMethod
    from art.defences.preprocessor import FeatureSqueezing

    model, probability_model, (x_train, y_train), (x_test, y_test) = setup_cifar10_model()
    x_train, x_test = x_train / 255, x_test / 255

    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3),
                                       clip_values=(0, 1), loss_object=model.loss)
    attack = FastGradientMethod(art_model, eps=0.1)
    result1 = attack.generate(x_test[:1000])

    model, probability_model, (x_train, y_train), (x_test, y_test) = setup_cifar10_model()
    x_train, x_test = x_train / 255, x_test / 255
    defence = FeatureSqueezing(clip_values=(0, 1), bit_depth=4)
    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3),
                                       clip_values=(0, 1), preprocessing_defences=defence, loss_object=model.loss)
    attack = FastGradientMethod(art_model, eps=0.1)
    result2 = attack.generate(x_test[:1000])
    a = 5


if __name__ == '__main__':
    # fgsm_adv_training()
    # squeeze()
    thermometer()
