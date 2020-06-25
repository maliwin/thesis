from util import *

preload_tensorflow()
import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical


def eval():
    from art.attacks.evasion import DeepFool
    from attacks.fgm import fgm
    from attacks.deepfool import deepfool

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    y_train, y_test = y_train.flatten(), y_test.flatten()

    model_final = tf.keras.models.load_model('distilation/model_final100_range01')
    model_clean = tf.keras.models.load_model('distilation/model_clean_range01')

    model_final.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    model_clean.compile(loss=tf.keras.losses.CategoricalCrossentropy())

    y_pred = np.argmax(model_clean.predict(x_test), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('acc on clean %.4f' % acc)

    y_pred = np.argmax(model_final.predict(x_test), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('acc on distilled %.4f' % acc)

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    art_model1 = TensorFlowV2Classifier(model_final, nb_classes=10, input_shape=(32, 32, 3),
                                        clip_values=(0, 1), loss_object=loss_object)
    a, b = fgm(art_model1, x_test, eps=2 / 255)
    y_pred = np.argmax(art_model1.predict(a), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('fgm eps2 on distilled %.4f' % acc)

    a, b = fgm(art_model1, x_test, eps=5 / 255)
    y_pred = np.argmax(art_model1.predict(a), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('fgm eps5 on distilled %.4f' % acc)

    a, b = fgm(art_model1, x_test, eps=10 / 255)
    y_pred = np.argmax(art_model1.predict(a), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('fgm eps10 on distilled %.4f' % acc)

    # p1 = []
    # for img in x_test[:500]:
    #     pimg, _ = deepfool(art_model1, np.array([img]), max_iter=20, eps=0.2)
    #     p1.append(pimg[0])
    # y_pred = np.argmax(art_model1.predict(np.array(p1)), axis=1)
    # acc = np.sum(y_pred == y_test[:500]) / len(y_test[:500])
    # print('acc on distilled deepfool %.4f' % acc)

    art_model2 = TensorFlowV2Classifier(model_clean, nb_classes=10, input_shape=(32, 32, 3),
                                        clip_values=(0, 1), loss_object=loss_object)
    a, b = fgm(art_model2, x_test, eps=2 / 255)
    y_pred = np.argmax(art_model2.predict(a), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('fgm eps2 on clean %.4f' % acc)

    a, b = fgm(art_model2, x_test, eps=5 / 255)
    y_pred = np.argmax(art_model2.predict(a), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('fgm eps5 on clean %.4f' % acc)

    a, b = fgm(art_model2, x_test, eps=10 / 255)
    y_pred = np.argmax(art_model2.predict(a), axis=1)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print('fgm eps10 on clean %.4f' % acc)

    # p2 = []
    # for img in x_test[:500]:
    #     pimg, _ = deepfool(art_model2, np.array([img]), max_iter=20, eps=0.2)
    #     p2.append(pimg[0])
    # y_pred = np.argmax(art_model2.predict(np.array(p2)), axis=1)
    # acc = np.sum(y_pred == y_test[:500]) / len(y_test[:500])
    # print('acc on clean deepfool %.4f' % acc)


def eval_carlini():
    from art.attacks.evasion import DeepFool
    from attacks.fgm import fgm
    from attacks.cw import cw_l2
    setup_logging()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    y_train, y_test = y_train.flatten(), y_test.flatten()

    model_final = tf.keras.models.load_model('distilation/model_final100_range01')
    model_clean = tf.keras.models.load_model('distilation/model_clean_range01')

    model_final.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    model_clean.compile(loss=tf.keras.losses.CategoricalCrossentropy())

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    art_model1 = TensorFlowV2Classifier(model_final, nb_classes=10, input_shape=(32, 32, 3),
                                        clip_values=(0, 1), loss_object=loss_object)
    art_model2 = TensorFlowV2Classifier(model_clean, nb_classes=10, input_shape=(32, 32, 3),
                                        clip_values=(0, 1), loss_object=loss_object)
    a, b = cw_l2(art_model2, x_test[:20], max_iter=10)
    y_pred = np.argmax(art_model2.predict(a), axis=1)
    acc = np.sum(y_pred == y_test[:20]) / len(y_test[:20])
    print(acc)


def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    y_train, y_test = y_train.flatten(), y_test.flatten()

    model, prob_model = get_untrained_model_tf()
    prob_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=['accuracy'])
    prob_model.fit(x_train, to_categorical(y_train, 10), epochs=50,
                   validation_data=(x_test, to_categorical(y_test, 10)))
    prob_model.save('distilation/model_clean_range01')

    lambda_layer = tf.keras.layers.Lambda(lambda x: x / 100)
    model, _ = get_untrained_model_tf()
    model_temp = tf.keras.Sequential([model, lambda_layer, tf.keras.layers.Softmax()])
    model_temp.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=['accuracy'])
    model_temp.fit(x_train, to_categorical(y_train, 10), epochs=50,
                   validation_data=(x_test, to_categorical(y_test, 10)), batch_size=128)

    model, _ = get_untrained_model_tf()
    model_distilled = tf.keras.Sequential([model, lambda_layer, tf.keras.layers.Softmax()])
    model_distilled.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.CategoricalCrossentropy(),
                            metrics=['accuracy'])
    y_pred = model_temp.predict(x_train)
    model_distilled.fit(x_train, y_pred, epochs=50, validation_data=(x_test, to_categorical(y_test, 10)),
                        batch_size=128)
    model_final = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    model_final.save('distilation/model_final100_range01')


if __name__ == '__main__':
    # setup_logging()
    # eval()
    eval_carlini()
    # train()
