from util import *
preload_tensorflow()

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    whichs = ['resnet50v2', 'xception']
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(whichs[0])
    predictions = decode_predictions(model.predict(preprocessed_images))
    for top5_preds in predictions:
        top1 = top5_preds[0]
        name, confidence = top1[1], top1[2] * 100
        print('%s %.2f%%' % (name.capitalize(), confidence))
