from util import *
preload_tensorflow()

import numpy as np

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
art_model = TensorFlowV2Classifier(model=model, nb_classes=1000, input_shape=(224, 224, 3), clip_values=(0, 1),  preprocessing=(1, 1))

target_image = Image.open('../../data/personal_images/test/test_im1.jpg')
target_image = target_image.resize((224, 224), resample=Image.LANCZOS)
target_image = np.array(target_image, dtype=np.float32) / 255
target_image_arr = np.array([target_image])

print(decode_predictions(art_model.predict(target_image_arr)))  # macaw

attack = DeepFool(classifier=art_model, max_iter=100, epsilon=0.02)

import time
t1 = time.time()
adv = attack.generate(x=target_image_arr)
t2 = time.time()
print(decode_predictions(art_model.predict(adv)))  # still macaw, failed to generate adversarial
print('Time spent %.3f' % (t2 - t1))
display_images(adv, (1, 1))

