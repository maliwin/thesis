import sys
sys.path.append('..')

from util import *
preload_tensorflow()
import tensorflow as tf
import keras
import keras.callbacks

from small_resnet.small_resnet_tf import resnet_v2
from art.utils import to_categorical
from art.classifiers import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD

setup_logging()

def adversarial_pgd():
    model = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    art_model = KerasClassifier(model, clip_values=(0, 255), preprocessing=(0, 1))
    trainer = AdversarialTrainerMadryPGD(art_model)
    import time
    t1 = time.time()
    cbs = [
        keras.callbacks.ModelCheckpoint(filepath='./checkpoints/ckpt4' + '/{epoch:02d}-{loss:.6f}',
                                        verbose=1, period=391),
    ]
    trainer.fit(x_train, to_categorical(y_train, 10), None, callbacks=cbs)
    t2 = time.time()
    print('time %f' % (t2 - t1))


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
    model1.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt4/01-1.366811')
    art_model1 = KerasClassifier(model1, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred1 = np.argmax(art_model1.predict(x_test), axis=1)
    i1 = np.where(y_pred1 == y_test)[0]

    model2 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model2.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt4/01-1.092752')
    art_model2 = KerasClassifier(model2, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred2 = np.argmax(art_model2.predict(x_test), axis=1)
    i2 = np.where(y_pred2 == y_test)[0]

    model3 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model3.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt4/01-1.011014')
    art_model3 = KerasClassifier(model3, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred3 = np.argmax(art_model3.predict(x_test), axis=1)
    i3 = np.where(y_pred3 == y_test)[0]

    model4 = resnet_v2(input_shape=(32, 32, 3), depth=3 * 6 + 2)
    model4.load_weights(ROOT_DIR + '/defenses/checkpoints/ckpt3_clean/16-0.5863')
    art_model4 = KerasClassifier(model4, clip_values=(0, 255), preprocessing=(0, 1))
    y_pred4 = np.argmax(art_model4.predict(x_test), axis=1)
    i4 = np.where(y_pred4 == y_test)[0]

    from art.attacks.evasion import CarliniL2Method, FastGradientMethod
    a1 = CarliniL2Method(art_model1, max_iter=100)
    # a2 = CarliniL2Method(art_model2, max_iter=10, learning_rate=0.3)
    # a3 = CarliniL2Method(art_model3, max_iter=10, learning_rate=0.3)
    a4 = CarliniL2Method(art_model4, max_iter=100)
    a1_gen = a1.generate(x_test[i1][:1])
    a4_gen = a4.generate(x_test[i4][:30])
    a = 5


if __name__ == '__main__':
    # adversarial_pgd()
    loaded()
    # regular_train()
