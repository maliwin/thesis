from util import *

preload_tensorflow()
setup_logging()

from art.attacks.evasion import ZooAttack


def zoo(art_model, images, labels, max_iter=10):
    attack = ZooAttack(art_model, max_iter=max_iter)
    adversarial_images = attack.generate(images, y=labels)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    images1, predictions = zoo(art_model, images[:1], correct_labels[:1], max_iter=2)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
