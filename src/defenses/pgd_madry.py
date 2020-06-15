import sys
sys.path.append('..')

from util import *
preload_tensorflow()
import tensorflow as tf
import keras
import keras.callbacks

from small_resnet import resnet_v2
from art.utils import to_categorical
from art.classifiers import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD

setup_logging()

def adversarial_pgd():
    # model = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    model = keras.models.load_model('./checkpoints/ckpt1/01-1.32')  # početak 391 - 72 - ~100 - 20 - 33
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    art_model = KerasClassifier(model, clip_values=(0, 255), preprocessing=(0, 1))
    trainer = AdversarialTrainerMadryPGD(art_model)
    import time
    t1 = time.time()
    cbs = [
        keras.callbacks.ModelCheckpoint(filepath='./checkpoints/ckpt1' + '/{epoch:02d}-{loss:.4f}',
                                        verbose=1, period=391),
    ]
    trainer.fit(x_train, to_categorical(y_train, 10), None, callbacks=cbs)
    t2 = time.time()
    print('time %f' % (t2 - t1))
    a = 5


def regular_train():
    model = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    cbs = [
        keras.callbacks.ModelCheckpoint(filepath='./checkpoints/ckpt3_clean' + '/{epoch:02d}-{loss:.4f}', save_best_only=True),
    ]
    model.fit(x_train, to_categorical(y_train, 10), epochs=50,
              validation_data=(x_test, to_categorical(y_test)), callbacks=cbs)
    model.save('saved_models/keras_cifar10_small_resnet_epochs50')
    a = 5


def loaded():
    from art.classifiers import KerasClassifier
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.flatten()
    x_test = x_test.astype(np.float32)

    model1 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model1.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt1/01-1.4025')
    art_model1 = KerasClassifier(model1, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred1 = np.argmax(art_model1.predict(x_test), axis=1)
    i1 = np.where(y_pred1 == y_test)[0]

    model2 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model2.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt1/01-0.97')
    art_model2 = KerasClassifier(model2, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred2 = np.argmax(art_model2.predict(x_test), axis=1)
    i2 = np.where(y_pred2 == y_test)[0]

    model3 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model3.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt1/01-1.0798')
    art_model3 = KerasClassifier(model3, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred3 = np.argmax(art_model3.predict(x_test), axis=1)
    i3 = np.where(y_pred3 == y_test)[0]

    model4 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model4.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt3_clean/16-0.5863')
    art_model4 = KerasClassifier(model4, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred4 = np.argmax(art_model4.predict(x_test), axis=1)
    i4 = np.where(y_pred4 == y_test)[0]

    from art.attacks.evasion import FastGradientMethod, CarliniL2Method
    a1 = CarliniL2Method(art_model1, max_iter=5, learning_rate=0.5)
    a2 = CarliniL2Method(art_model2, max_iter=5, learning_rate=0.5)
    a3 = CarliniL2Method(art_model3, max_iter=5, learning_rate=0.5)
    a4 = CarliniL2Method(art_model4, max_iter=5, learning_rate=0.5)
    a1.generate(x_test[i1][:10])
    a2.generate(x_test[i2][:10])
    a3.generate(x_test[i3][:10])
    a4.generate(x_test[i4][:10])
    a = 5


if __name__ == '__main__':
    # adversarial_pgd()
    loaded()
    # regular_train()

