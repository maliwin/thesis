from util import *

preload_tensorflow()

from art.attacks.evasion import ProjectedGradientDescent


def pgd(art_model, images, eps, norm=np.inf, max_iter=100, eps_step=0.1):
    attack = ProjectedGradientDescent(art_model, norm=norm, eps=eps, max_iter=max_iter, eps_step=eps_step)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    setup_logging()
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    # note: good inf norm epsilons: 1, 5, 10
    images1, predictions = pgd(art_model, images, eps=1, max_iter=10)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)

    display_images(adv[-3:], (1, 3))

    for a in images1:
        z = np.array([a])
        print(decode_predictions(art_model.predict(z)))
    print(decode_predictions(art_model.predict(not_adv)))
