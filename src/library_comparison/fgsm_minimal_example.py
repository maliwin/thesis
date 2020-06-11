from util import *
preload_tensorflow()
setup_logging()

from art.classifiers import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod


def fgm_255():
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    model = ResNet50V2()
    art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                       nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 255),
                                       preprocessing=(127.5, 127.5))
    images, labels = load_personal_images()[2:3]
    images = images.astype(np.float32)

    attack = FastGradientMethod(art_model, eps=10)
    adversarial_images = attack.generate(images)
    print('inf norm: %f' % np.abs((adversarial_images - images)).max())


def fgm_1():
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    model = ResNet50V2()
    art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                       nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 1),
                                       preprocessing=(0.5, 0.5))
    images = load_personal_images()[2:3]
    images = images / 255

    attack = FastGradientMethod(art_model, eps=10 / 255)
    adversarial_images = attack.generate(images)
    print('inf norm: %f' % np.abs((adversarial_images - images)).max())


if __name__ == '__main__':
    fgm_255()
    fgm_1()
