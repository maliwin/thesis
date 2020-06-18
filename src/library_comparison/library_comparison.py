import sys
sys.path.append('..')

import time
import tensorflow as tf
from util import *
preload_tensorflow()


def foolbox_example():
    from foolbox import TensorFlowModel
    from foolbox.attacks import FGSM

    def _foolbox_example_tf():
        model, probability_model, (x_train, y_train), (x_test, y_test) = setup_cifar10_model()

        # NB: "The size of the perturbations should be at most epsilon, but this
        #     is not guaranteed and the caller should verify this or clip the result"
        # epsilons = [0.0001, 0.001, 0.01, 0.02, 0.03]
        epsilons = [0.01]
        # x_test, y_test = tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test.flatten())
        # x_test, y_test = ep.astensors(*samples(fmodel, dataset='cifar10', batchsize=4))
        # x_test, y_test = ep.astensors(x_test[:4], y_test[:4])
        # accuracy(fmodel, x_test, y_test)

        # foolbox requires a very specific dataformat
        x_test_converted, y_test_converted = tf.convert_to_tensor(x_test[:4]),\
                                             tf.convert_to_tensor(y_test[:4].flatten().astype(np.int32))

        t1 = time.time()
        fmodel = TensorFlowModel(model, bounds=(0, 1))
        attack = FGSM()
        advs, advs_clipped, success = attack(fmodel, x_test_converted, y_test_converted, epsilons=epsilons)
        print('Foolbox time: ' + str(time.time() - t1))

        images = []
        class_ids = []
        for adv in advs[0]:
            t = adv.numpy().reshape(1, 32, 32, 3)
            class_id = np.argmax(probability_model.predict(t))
            images.append(adv.numpy())
            class_ids.append(class_id)

        display_images(images, (2, 2), titles=cifar10_class_id_to_text(class_ids))

    _foolbox_example_tf()


def cleverhans_example():
    from cleverhans.future.tf2.attacks import fast_gradient_method

    def _cleverhans_example_tf():
        model, probability_model, (x_train, y_train), (x_test, y_test) = setup_cifar10_model()

        t1 = time.time()
        adversarial = fast_gradient_method(model, x_test[:4], 0.01, np.inf)
        print('Cleverhans time: ' + str(time.time() - t1))

        images = []
        class_ids = []

        for adv in adversarial:
            t = adv.numpy().reshape(1, 32, 32, 3)
            class_id = np.argmax(probability_model.predict(t))
            images.append(adv.numpy())
            class_ids.append(class_id)

        display_images(images, (2, 2), titles=cifar10_class_id_to_text(class_ids))

    _cleverhans_example_tf()


def art_example():
    from art.attacks.evasion import FastGradientMethod
    from art.classifiers import TensorFlowV2Classifier

    def _art_example_tf():
        model, probability_model, (x_train, y_train), (x_test, y_test) = setup_cifar10_model()

        t1 = time.time()
        model_art = TensorFlowV2Classifier(model=probability_model, loss_object=model.loss,
                                           nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1))
        attack = FastGradientMethod(model_art, eps=0.01)
        adversarials = attack.generate(x=np.array([x_test[0]]), x_adv_init=None)
        print('ART time: ' + str(time.time() - t1))

        images = []
        class_ids = []

        for adv in adversarials:
            t = adv.reshape(1, 32, 32, 3)
            class_id = np.argmax(probability_model.predict(t))
            images.append(adv)
            class_ids.append(class_id)

        display_images(images, (2, 2), titles=cifar10_class_id_to_text(class_ids))

    _art_example_tf()


if __name__ == '__main__':
    # foolbox_example()
    # cleverhans_example()
    art_example()
