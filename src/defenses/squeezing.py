import sys
import time

from util import *
preload_tensorflow()
import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical

setup_logging()

from art.attacks.evasion import DeepFool
from art.defences.preprocessor import SpatialSmoothing, FeatureSqueezing


def squeeze_display():
    img = load_personal_images((128, 128))[0][4]
    squozen = []
    for bits in range(1, 9, 2):
        defence = FeatureSqueezing(clip_values=(0, 255), bit_depth=(9 - bits))
        print(9 - bits)
        sq, _ = defence(img)
        squozen.append(sq)
    defence = FeatureSqueezing(clip_values=(0, 255), bit_depth=1)
    sq, _ = defence(img)
    squozen.append(sq)
    display_images(squozen, (1, 5))


def squeeze():
    model, art_model, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(classifier_activation=None)

    x, y = get_some_imagenet_set()
    x = x.astype(np.float32) / 255
    y = y

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    y_pred = np.argmax(model.predict((x - 0.5) / 0.5), axis=1)
    acc = np.sum(y_pred == y) / len(y)
    print('clean acc %.4f' % acc)

    art_model = TensorFlowV2Classifier(model, nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 1),
                                       loss_object=loss_object, preprocessing=(0.5, 0.5))

    from attacks.fgm import fgm
    a1, _ = fgm(art_model, x, eps=0.3)

    y_pred = np.argmax(art_model.predict(a1), axis=1)
    acc = np.sum(y_pred == y) / len(y)
    print('no defence acc %.4f' % acc)

    defence = SpatialSmoothing()
    art_model = TensorFlowV2Classifier(model, nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 1),
                                       loss_object=loss_object, preprocessing_defences=defence, preprocessing=(0.5, 0.5))

    y_pred = np.argmax(art_model.predict(a1), axis=1)
    acc = np.sum(y_pred == y) / len(y)
    print('defence acc on premade %.4f' % acc)

    a1, _ = fgm(art_model, x, eps=0.3)

    y_pred = np.argmax(art_model.predict(a1), axis=1)
    acc = np.sum(y_pred == y) / len(y)
    print('defence acc on postmade %.4f' % acc)


if __name__ == '__main__':
    # squeeze_display()
    squeeze()

