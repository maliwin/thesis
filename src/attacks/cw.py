from util import *

preload_tensorflow()
setup_logging()

from art.attacks.evasion import CarliniLInfMethod, CarliniL2Method


def cw_l2(art_model, images, max_iter=5):
    attack = CarliniL2Method(art_model, max_iter=max_iter, learning_rate=0.02)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


def cw_linf(art_model, images):
    attack = CarliniLInfMethod(art_model, max_iter=120, learning_rate=0.5, eps=5)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


def cw_false_negative(art_model, n=1, confidence=0.95, max_iter=20, learning_rate=0.5, *args):
    img = np.array([np.random.random((224, 224, 3)) * 255 for _ in range(n)])
    attack = CarliniL2Method(art_model, confidence=confidence, max_iter=max_iter, learning_rate=learning_rate)
    adversarial_images = attack.generate(img)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    images1, predictions = cw_l2(art_model, images)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    display_images(adv, (4, 4))
