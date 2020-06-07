import sys
import time

from util import *
preload_tensorflow()
import tensorflow as tf
from art.classifiers import TensorFlowV2Classifier
from art.utils import to_categorical

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# napadi:
# FG(S)M https://arxiv.org/abs/1412.6572 (rand-fgsm?)
# JSMA (weak?)
# DeepFool
# C&W
# Pixel Attack
# Boundary attack
# HopSkipJump attack


def setup(which='resnet50v2'):
    images = load_images('../data/personal_images', (224, 224))
    images = images.astype(np.float64)

    def _setup_resnetv2_50():
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
        model = ResNet50V2()
        art_preprocessing = preprocessing_art_tuple('tf')
        art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                           nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 255),
                                           preprocessing=art_preprocessing)
        preprocessed_images = preprocess_input(np.array(images))
        return model, art_model, images, preprocessed_images, preprocess_input, decode_predictions

    def _setup_mobilenetv2():
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
        model = MobileNetV2()
        art_preprocessing = preprocessing_art_tuple('tf')
        art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                           nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 255),
                                           preprocessing=art_preprocessing)
        preprocessed_images = preprocess_input(np.array(images))
        return model, art_model, images, preprocessed_images, preprocess_input, decode_predictions

    def _setup_densenet_121():
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
        model = DenseNet121()
        art_preprocessing = preprocessing_art_tuple('torch')
        art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                           nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 255),
                                           preprocessing=art_preprocessing)
        preprocessed_images = preprocess_input(np.array(images))
        return model, art_model, images, preprocessed_images, preprocess_input, decode_predictions

    def _setup_vgg16():
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        # NB: VGG16 is BGR
        # NB: preprocess_input converts RGB to BGR on its own ! ! !
        #     however, art DOESN'T and we have to pass it a BGR image that has *not* been preprocessed
        #     (because it does it on it's own too)

        model = VGG16()
        art_preprocessing = preprocessing_art_tuple('caffe')
        art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                           nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 255),
                                           preprocessing=art_preprocessing)
        preprocessed_images = preprocess_input(np.array(images))  # these are now bgr !!!!!!!!!
        return model, art_model, images, preprocessed_images, preprocess_input, decode_predictions

    def _setup_vgg19():
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
        # NB: VGG16 is BGR
        # NB: preprocess_input converts RGB to BGR on its own ! ! !
        #     however, art DOESN'T and we have to pass it a BGR image that has *not* been preprocessed
        #     (because it does it on it's own too)

        model = VGG19()
        art_preprocessing = preprocessing_art_tuple('caffe')
        art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                           nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 255),
                                           preprocessing=art_preprocessing)
        preprocessed_images = preprocess_input(np.array(images))  # these are now bgr !!!!!!!!!
        return model, art_model, images, preprocessed_images, preprocess_input, decode_predictions

    def _setup_xception_deprecated():
        # deprecated because it's 299, it's easier to just do 224 on all of them
        from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
        model = Xception()
        art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                           nb_classes=1000, input_shape=(299, 299, 3), clip_values=(0, 255),
                                           preprocessing=(127.5, 127.5))
        preprocessed_images = preprocess_input(np.array(images))
        return model, art_model, images, preprocessed_images, preprocess_input, decode_predictions

    lookup_setup = {
        'resnet50v2': _setup_resnetv2_50,
        'densenet121': _setup_densenet_121,
        'vgg16': _setup_vgg16,
        'vgg19': _setup_vgg19,
        'mobilenetv2': _setup_mobilenetv2,
        'xception': _setup_xception_deprecated
    }

    return lookup_setup[which]()


def fgsm():
    # FGSM: norms - 1, 2, inf
    from art.attacks.evasion import FastGradientMethod
    # NB: np.inf is FGSM
    model, art_model, images, preprocessed_images, preprocess_input, decode_predictions = setup()

    # NB: epsilon depends on input, i.e. if images are [0, 1] then eps should be of the same order of magnitude
    #     if images are [0, 255], then eps has to be on that order of magnitude
    norms = [np.inf, 1, 2]
    epsilons = [0.5, 1, 2, 5, 10, 20]
    adversarials = [[] for _ in range(len(norms))]  # one list for every norm
    predictions = [[] for _ in range(len(norms))]

    for idx, norm in enumerate(norms):
        for eps in epsilons:
            attack = FastGradientMethod(art_model, norm=norm, minimal=True, eps=eps)
            adversarial_images = attack.generate(images)
            adversarial_predictions = decode_predictions(art_model.predict(adversarial_images))
            adversarials[idx].append(adversarial_images)
            predictions[idx].append(adversarial_predictions)
    # TODO: get labels
    a = 5


def jsma():
    from art.attacks.evasion import SaliencyMapMethod
    model, art_model, images, preprocessed_images, preprocess_input, decode_predictions = setup()

    attack = SaliencyMapMethod(art_model)
    adversarial_images = attack.generate(np.array([images[0]]))
    adversarial_predictions = decode_predictions(model.predict(adversarial_images))
    a = 5


def pgd():
    from art.attacks.evasion import ProjectedGradientDescent
    model, art_model, images, preprocessed_images, preprocess_input, decode_predictions = setup()

    attack = ProjectedGradientDescent(art_model, max_iter=10)
    adversarial_images = attack.generate(images)
    adversarial_predictions = decode_predictions(art_model.predict(adversarial_images))
    orig_predictions = decode_predictions(model.predict(preprocessed_images))


def cw():
    from art.attacks.evasion import CarliniL2Method, CarliniLInfMethod
    model, art_model, images, preprocessed_images, preprocess_input, decode_predictions = setup()

    # attack1 = CarliniL2Method(art_model, max_iter=30)
    # adversarial_images1 = attack1.generate(np.array([images[0]]))
    # adversarial_predictions1 = decode_predictions(art_model.predict(adversarial_images1))

    attack2 = CarliniLInfMethod(art_model, max_iter=0, learning_rate=0.2, eps=1.2)
    adversarial_images2 = attack2.generate(images)
    display_images(adversarial_images2 / 255, (4, 4))
    adversarial_predictions2 = decode_predictions(art_model.predict(adversarial_images2))
    a = 5


def boundary_attack(init_image_idxs=None, target_image_idxs=None,
                    targeted=False, which_model='resnet50v2',
                    iter_step=200, iter_count=20, bgr=False,
                    callback=None):
    import time
    from art.attacks.evasion import BoundaryAttack
    model, art_model, images, preprocessed_images, preprocess_input, decode_predictions = setup(which_model)

    if init_image_idxs or target_image_idxs:
        assert len(init_image_idxs) == len(target_image_idxs)

    # become a dragonfly (319)
    target_images = np.array(images[target_image_idxs]) if target_image_idxs else np.array(images)
    target_images_preprocessed = preprocess_input(np.array(target_images))
    target_images_predictions = np.argmax(model.predict(target_images_preprocessed), axis=1)

    # start from llama (355)
    init_images = np.array(images[init_image_idxs]) if init_image_idxs else np.array(images)
    init_images_preprocessed = preprocess_input(np.array(init_images))  # if bgr, preprocess will handle RGB -> BGR
    init_images_predictions = np.argmax(model.predict(init_images_preprocessed), axis=1)

    if bgr:
        target_images = target_images[..., ::-1]
        init_images = init_images[..., ::-1]

    attack = BoundaryAttack(art_model, targeted=targeted, max_iter=0)
    max_iter = iter_step * (iter_count + 1)
    x_adv = init_images if targeted else None

    x_advs = []
    predictions = []

    t1 = time.time()
    for i in range(max_iter // iter_step):
        y = tf.one_hot(init_images_predictions, 1000) if targeted else None
        x_adv = attack.generate(x=target_images, y=y, x_adv_init=x_adv)
        prediction = decode_predictions(art_model.predict(x_adv))

        x_advs.append(x_adv)
        predictions.append(prediction)
        print('Iteration %d at time %f' % (i * 200, time.time() - t1))

        # this stuff is for the first image only
        l2_error = np.linalg.norm(np.reshape(x_adv[0] - target_images, [-1]))
        single_prediction_class, single_prediction_prob = prediction[0][0][1:]
        print('L2 error: %f Prediction: %s %f' % (l2_error, single_prediction_class, single_prediction_prob))

        attack.max_iter = iter_step
        attack.delta = attack.curr_delta
        attack.epsilon = attack.curr_epsilon

        if callback:
            callback(x_adv)

    return np.array(x_advs), np.array(predictions)


if __name__ == '__main__':
    # fgsm()
    # jsma()
    # deepfool()
    pgd()
    # cw()
    # boundary_attack(which_model='densenet121')
