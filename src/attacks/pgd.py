from util import *

preload_tensorflow()
setup_logging()

from art.attacks.evasion import ProjectedGradientDescent


def pgd(art_model, images, eps, norm=np.inf, max_iter=100):
    attack = ProjectedGradientDescent(art_model, norm=norm, eps=eps, max_iter=max_iter)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    # note: good inf norm epsilons: 1, 5, 10
    images1, predictions = pgd(art_model, images, eps=100)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
