from util import *

preload_tensorflow()
setup_logging()

import time
import numpy as np
from art.attacks.evasion import BoundaryAttack


def boundary_attack(art_model, target_images, init_images=None,
                    iter_step=200, iter_count=20, num_classes=1000, callback=None):
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

    # become a dragonfly (319)
    # start from llama (355)

    attack = BoundaryAttack(art_model, targeted=targeted, max_iter=0)
    max_iter = iter_step * (iter_count + 1)
    x_adv = init_images if targeted else None

    x_advs = []

    t1 = time.time()
    for i in range(max_iter // iter_step):
        y = tf.one_hot(np.argmax(art_model.predict(init_images), axis=1), num_classes) if targeted else None
        x_adv = attack.generate(x=target_images, y=y, x_adv_init=x_adv)

        x_advs.append(x_adv)
        print('Iteration %d at time %f' % (i * iter_step, time.time() - t1))

        # this stuff is for the first image only
        l2_error = np.linalg.norm(np.reshape(x_adv[0] - target_images, [-1]))
        print('L2 error: %f' % l2_error)

        attack.max_iter = iter_step
        attack.delta = attack.curr_delta
        attack.epsilon = attack.curr_epsilon

        if callback:
            callback(x_adv)

    return np.array(x_advs)


if __name__ == '__main__':
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    adv_img_history, predictions = boundary_attack(art_model, images[2], images[5], iter_count=1, iter_step=10)
    y_pred = np.argmax(predictions, axis=1)
    # display_images(adv_img_history, (2, 3))
