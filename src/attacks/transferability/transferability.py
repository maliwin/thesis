from util import *
preload_tensorflow()

from art.classifiers import TensorFlowV2Classifier
from attacks.fgm import fgm
from attacks.cw import cw_linf, cw_l2


def make_adv_examples(which):
    assert which in ('A', 'B')
    model, _, (x_train, y_train), (x_test, y_test) = setup_cifar10_model(50)
    if which == 'B':
        model = tf.keras.models.load_model('cifar10_other_model_2')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                       loss_object=loss_object)

    # adv, _ = fgm(art_model, x_test, eps=2/255)
    # print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    # save_numpy_array(adv, 'fgsm_%s_2' % which, '.')
    #
    # adv, _ = fgm(art_model, x_test, eps=5/255)
    # print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    # save_numpy_array(adv, 'fgsm_%s_5' % which, '.')
    #
    # adv, _ = fgm(art_model, x_test, eps=10/255)
    # print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    # save_numpy_array(adv, 'fgsm_%s_10' % which, '.')

    # advs = []
    # for img in x_test[:200]:
    #     adv, _ = cw_l2(art_model, np.array([img]), max_iter=20, confidence=50)
    #     adv = adv[0]
    #     advs.append(adv)
    # advs = np.array(advs)
    # save_numpy_array(advs, 'carlini_l2_%s_conf50_200' % which, '.')
    # print(np.sum(np.argmax(art_model.predict(advs), axis=1) == y_test[:200].flatten()) / len(y_test[:200].flatten()))

    advs = []
    for img in x_test[:500]:
        adv, _ = cw_linf(art_model, np.array([img]), confidence=50, max_iter=10, eps=10/255)
        adv = adv[0]
        advs.append(adv)
    advs = np.array(advs)
    # save_numpy_array(advs, 'carlini_linf_%s_conf50_eps10_500' % which, '.')
    print(np.sum(np.argmax(art_model.predict(advs), axis=1) == y_test[:500].flatten()) / len(y_test[:500].flatten()))

def train_some_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range
    y_test = y_test.flatten()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))
    model.save('cifar10_other_model_2')


def test_adv_examples():
    model_a, _, (x_train, y_train), (x_test, y_test) = setup_cifar10_model(50)
    model_b = tf.keras.models.load_model('cifar10_other_model_2')

    def test_model(model, name):
        print('- - TESTING MODEL %s - -' % name)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=loss,
                      metrics=['accuracy'])

        art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                           loss_object=loss)

        print('Clean accuracy')
        print(np.sum(np.argmax(art_model.predict(x_test), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv1 = load_numpy_array('fgsm_B_2', '.')
        print('FGSM 2 from model B')
        print(np.sum(np.argmax(art_model.predict(adv1), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv2 = load_numpy_array('fgsm_B_5', '.')
        print('FGSM 5 from model B')
        print(np.sum(np.argmax(art_model.predict(adv2), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv3 = load_numpy_array('fgsm_B_10', '.')
        print('FGSM 10 from model B')
        print(np.sum(np.argmax(art_model.predict(adv3), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv4 = load_numpy_array('carlini_linf_B_conf50_eps10_500', '.')
        print('Linf 10 from model B, adv then clean')
        print(np.sum(np.argmax(art_model.predict(adv4), axis=1) == y_test[:500].flatten()) / len(y_test[:500].flatten()))
        print(np.sum(np.argmax(art_model.predict(x_test[:500]), axis=1) == y_test[:500].flatten()) / len(y_test[:500].flatten()))

        adv5 = load_numpy_array('carlini_l2_B_conf50_200', '.')
        print('L2 from model B, adv then clean')
        print(np.sum(np.argmax(art_model.predict(adv5), axis=1) == y_test[:200].flatten()) / len(y_test[:200].flatten()))
        print(np.sum(np.argmax(art_model.predict(x_test[:200]), axis=1) == y_test[:200].flatten()) / len(y_test[:200].flatten()))

        print('-'*30)

        adv = load_numpy_array('fgsm_A_2', '.')
        print('FGSM 2 from model A')
        print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv = load_numpy_array('fgsm_A_5', '.')
        print('FGSM 5 from model A')
        print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv = load_numpy_array('fgsm_A_10', '.')
        print('FGSM 10 from model A')
        print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test.flatten()) / len(y_test.flatten()))

        adv = load_numpy_array('carlini_linf_A_conf50_eps10_500', '.')
        print('Linf 10 from model A, adv then clean')
        print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test[:500].flatten()) / len(y_test[:500].flatten()))
        print(np.sum(np.argmax(art_model.predict(x_test[:500]), axis=1) == y_test[:500].flatten()) / len(
            y_test[:500].flatten()))

        adv = load_numpy_array('carlini_l2_A_conf50_200', '.')
        print('L2 from model A, adv then clean')
        print(np.sum(np.argmax(art_model.predict(adv), axis=1) == y_test[:200].flatten()) / len(y_test[:200].flatten()))
        print(np.sum(np.argmax(art_model.predict(x_test[:200]), axis=1) == y_test[:200].flatten()) / len(
            y_test[:200].flatten()))

    test_model(model_b, 'B')
    test_model(model_a, 'A')


def test_just_noise():
    model_a, _, (x_train, y_train), (x_test, y_test) = setup_cifar10_model(50)
    model_b = tf.keras.models.load_model('cifar10_other_model_2')

    print('Clean accuracy A')
    print(np.sum(np.argmax(model_a.predict(x_test), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    print('Noisy accuracy A')
    noise1 = np.where(np.random.random((10000, 32, 32, 3)) < 0.5, -10, 10) / 255
    x_test_noisyA = np.clip(x_test + noise1, 0, 1)
    print(np.sum(np.argmax(model_a.predict(x_test_noisyA), axis=1) == y_test.flatten()) / len(y_test.flatten()))

    print('Clean accuracy B')
    print(np.sum(np.argmax(model_b.predict(x_test), axis=1) == y_test.flatten()) / len(y_test.flatten()))
    print('Noisy accuracy B')
    noise2 = np.where(np.random.random((10000, 32, 32, 3)) < 0.5, -10, 10) / 255
    x_test_noisyB = np.clip(x_test + noise2, 0, 1)
    print(np.sum(np.argmax(model_a.predict(x_test_noisyB), axis=1) == y_test.flatten()) / len(y_test.flatten()))


if __name__ == '__main__':
    setup_logging()
    # train_some_model()
    # test_just_noise()
    # make_adv_examples('AB')
    test_adv_examples()
