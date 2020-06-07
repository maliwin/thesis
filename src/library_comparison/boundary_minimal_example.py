from util import *

import numpy as np
import tensorflow as tf
preload_tensorflow()

from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.applications.xception import Xception, decode_predictions, preprocess_input
from art.classifiers import TensorFlowV2Classifier
from art.attacks.evasion import BoundaryAttack

target_image = Image.open('../../data/personal_images/boundary_test/dragonfly/dragonfly.jpg')
target_image = target_image.resize((299, 299), resample=Image.LANCZOS)
target_image = np.array(target_image, dtype=np.float64)

model = Xception(weights='imagenet')
art_model = TensorFlowV2Classifier(model=model, nb_classes=1000, input_shape=(299, 299, 3), clip_values=(0, 255),
                                   preprocessing=(127.5, 127.5))
attack = BoundaryAttack(estimator=art_model, targeted=False, max_iter=0, delta=0.001, epsilon=0.001)

print('class id: ' + str(np.argmax(art_model.predict(np.array([target_image])))))
print(decode_predictions(art_model.predict(np.array([target_image]))))  # correctly classified as dragonfly

iter_step = 200
x_adv = None
x_advs = []
predictions = []

for i in range(20):
    x_adv = attack.generate(x=np.array([target_image]), x_adv_init=x_adv)
    prediction = decode_predictions(model.predict(x_adv))
    x_advs.append(x_adv)
    predictions.append(prediction)

    print("Adversarial image at step %d." % (i * iter_step), "L2 error",
          np.linalg.norm(np.reshape(x_adv[0] - target_image, [-1])),
          "and class label %d." % np.argmax(art_model.predict(x_adv)[0]))

    plt.imshow(x_adv[0].astype(np.uint))
    plt.show(blocking=False)

    if hasattr(attack, 'curr_delta') and hasattr(attack, 'curr_epsilon'):
        attack.max_iter = iter_step
        attack.delta = attack.curr_delta
        attack.epsilon = attack.curr_epsilon
    else:
        break
a = 5
