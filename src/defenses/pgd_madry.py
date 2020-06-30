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
from art.defences.trainer import AdversarialTrainerMadryPGD, AdversarialTrainerFBFPyTorch


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
    model1.load_weights(ROOT_DIR + '/defenses/checkpoints/final/madry2-1.366811')
    art_model1 = KerasClassifier(model1, clip_values=(0, 255), preprocessing=(0, 1))
    acc = np.sum(np.argmax(art_model1.predict(x_test), axis=1) == y_test) / len(y_test)
    print('Clean acc on pgd trained adversary: %.4f' % acc)

    from art.attacks.evasion import CarliniL2Method, FastGradientMethod, DeepFool, CarliniLInfMethod, ProjectedGradientDescent
    a1 = FastGradientMethod(art_model1, eps=8)
    t1 = a1.generate(x_test)
    a5 = ProjectedGradientDescent(art_model1, eps=8)
    t5 = a5.generate(x_test)
    a2 = DeepFool(art_model1, max_iter=10)
    t2 = a2.generate(x_test[:100])
    a3 = CarliniLInfMethod(art_model1, max_iter=100, eps=5)
    t3 = a3.generate(x_test[:100])
    a4 = CarliniL2Method(art_model1, max_iter=100)
    t4 = a4.generate(x_test[:100])
    a = 5


if __name__ == '__main__':
    setup_logging()
    # adversarial_pgd()
    loaded()
    # regular_train()
