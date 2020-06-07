import sys
import time
sys.path.append('..')

import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model
import keras.backend as k
from matplotlib import pyplot as plt

from art.classifiers import KerasClassifier
from art.attacks import BoundaryAttack
from art.utils import to_categorical

from util import *

mean_imagenet = np.zeros([224, 224, 3])
mean_imagenet[..., 0].fill(103.939)
mean_imagenet[..., 1].fill(116.779)
mean_imagenet[..., 2].fill(123.68)
model = ResNet50(weights='imagenet')
classifier = KerasClassifier(clip_values=(0, 255), model=model, preprocessing=(mean_imagenet, 1))

images = load_images('../../data/personal_images/boundary_test', (224, 224))
target_image, init_image = images.astype(np.float64)
koala_prediction = np.argmax(classifier.predict(np.array([target_image])))
tractor_prediction = np.argmax(classifier.predict(np.array([init_image])))

attack = BoundaryAttack(classifier=classifier, targeted=False, max_iter=0, delta=0.001, epsilon=0.001)
iter_step = 200
x_adv = None
for i in range(20):
    x_adv = attack.generate(x=np.array([target_image]), x_adv_init=x_adv)

    # clear_output()
    print("Adversarial image at step %d." % (i * iter_step), "L2 error",
          np.linalg.norm(np.reshape(x_adv[0] - target_image, [-1])),
          "and class label %d." % np.argmax(classifier.predict(x_adv)[0]))
    plt.imshow(x_adv[0].astype(np.uint))
    plt.show(block=False)

    if hasattr(attack, 'curr_delta') and hasattr(attack, 'curr_epsilon'):
        attack.max_iter = iter_step
        attack.delta = attack.curr_delta
        attack.epsilon = attack.curr_epsilon
    else:
        break

a = 5
