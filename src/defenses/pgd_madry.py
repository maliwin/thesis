from util import *
preload_tensorflow()
import tensorflow as tf
import keras
import keras.callbacks

from test import resnet_v2
from art.utils import to_categorical
from art.classifiers import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD


def adversarial_pgd():
    # model = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    model = keras.models.load_model('./checkpoints/ckpt1/01-1.32')
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    art_model = KerasClassifier(model, clip_values=(0, 255), preprocessing=(0, 1))
    trainer = AdversarialTrainerMadryPGD(art_model)
    import time
    t1 = time.time()
    cbs = [
        keras.callbacks.ModelCheckpoint(filepath='./checkpoints/ckpt1' + '/{epoch:02d}-{loss:.2f}',
                                        verbose=1, period=391),
    ]
    trainer.fit(x_train, to_categorical(y_train, 10), None, callbacks=cbs)
    t2 = time.time()
    print('time %f' % (t2 - t1))
    a = 5


def loaded():
    model = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)
    model.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt2/01-1.97')
    a = 5


if __name__ == '__main__':
    adversarial_pgd()
    # loaded()

