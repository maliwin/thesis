from util import *
preload_tensorflow()
import numpy as np
import time

from PIL import Image
from art.classifiers import TensorFlowV2Classifier
from art.attacks.evasion import DeepFool
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

model = ResNet50V2(weights='imagenet', classifier_activation=None)
art_model = TensorFlowV2Classifier(model=model, nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 1), preprocessing=(0.5, 0.5))

img1 = Image.open('../../data/personal_images/test/test_im1.jpg')
img1 = np.array(img1.resize((224, 224), resample=Image.LANCZOS), dtype=np.float32) / 255
img2 = Image.open('../../data/personal_images/test/test_im2.jpg')
img2 = np.array(img2.resize((224, 224), resample=Image.LANCZOS), dtype=np.float32) / 255
img1_arr, img2_arr = np.array([img1]), np.array([img2])
img_arr = np.array([img1, img2])

print(decode_predictions(model.predict(img1_arr)))  # macaw
print(decode_predictions(model.predict(img2_arr)))  # horse cart

attack = DeepFool(classifier=art_model, epsilon=1e-6, nb_grads=5, max_iter=20)

t1 = time.time()
adv = attack.generate(x=img1_arr)
print(decode_predictions(art_model.predict(adv)))  # hornbill, success after 5 iterations / ~2 seconds
t2 = time.time()
print('Time spent for img1 %.3f' % (t2 - t1))

t1 = time.time()
adv = attack.generate(x=img2_arr)
print(decode_predictions(art_model.predict(adv)))  # arabian camel, success after 2 iterations / ~1 second
t2 = time.time()
print('Time spent for img2 %.3f' % (t2 - t1))

t1 = time.time()
adv = attack.generate(x=img_arr)  # this will go through max_iter, in this case 20 iterations
print(decode_predictions(art_model.predict(adv)))  # [macaw, horse cart(!!)], unsuccessful attack for img2 / ~24 seconds
t2 = time.time()
print('Time spent for [img1, img2] %.3f' % (t2 - t1))
