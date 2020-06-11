from util import *

preload_tensorflow()
setup_logging()

from art.attacks.evasion import FastGradientMethod


def fgm(art_model, images, eps, norm=np.inf, minimal=False):
    # NB: epsilon depends on input, i.e. if images are [0, 1] then eps should be of the same order of magnitude
    #     if images are [0, 255], then eps has to be on that order of magnitude
    attack = FastGradientMethod(art_model, norm=norm, eps=eps, minimal=minimal)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    # note: good inf norm epsilons: 1, 5, 10
    # images1, predictions = fgm(art_model, images, eps=70)
    # y_pred = np.argmax(predictions, axis=1)
    # adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)

    # note: good 1 norm epsilons: 30k, probably don't need more
    # images1, predictions = fgm(art_model, images, norm=1, eps=300000)
    # y_pred = np.argmax(predictions, axis=1)
    # adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    # display_images(adv, (4, 4))

    # note: good 2 norm epsilons: 500, 2000
    images1, predictions = fgm(art_model, images, norm=2, eps=2000)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    display_images(adv, (4, 4))

    a = 5
    # save_images_plus_arrays(adv, subdirectory='fgm/norm_inf/eps_10', name_prefix='adv')
