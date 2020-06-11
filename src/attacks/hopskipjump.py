from util import *

preload_tensorflow()
setup_logging()

import time
import numpy as np
from art.attacks.evasion import HopSkipJump


def hopskipjump(art_model, target_images, init_images=None, max_iter=50, num_classes=1000):
    if init_images is not None:
        if init_images.ndim == 3 and target_images.ndim == 4:
            init_images = np.array([init_images] * len(target_images))
        if target_images.ndim == 3 and target_images.ndim == 3:
            init_images = np.array([init_images])
            target_images = np.array([target_images])
        if init_images.ndim == 4 and target_images.ndim == 3:
            target_images = np.array([target_images] * len(init_images))
        assert len(target_images) == len(init_images)
    else:
        if target_images.ndim == 3:
            target_images = np.array([target_images])

    if init_images is not None:
        targeted = True
    else:
        targeted = False

    attack = HopSkipJump(art_model, targeted=targeted, max_iter=max_iter)
    x_adv = init_images if targeted else None

    y = tf.one_hot(np.argmax(art_model.predict(init_images), axis=1), num_classes) if targeted else None
    x_adv = attack.generate(x=target_images, y=y, x_adv_init=x_adv)
    return x_adv, art_model.predict(x_adv)


if __name__ == '__main__':
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    adversarials, predictions = hopskipjump(art_model, images[2], images[5], max_iter=5)
    y_pred = np.argmax(predictions, axis=1)
