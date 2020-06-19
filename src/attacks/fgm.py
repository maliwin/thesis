from util import *

preload_tensorflow()
setup_logging()

from art.attacks.evasion import FastGradientMethod


def fgm(art_model, images, eps, norm=np.inf, minimal=False):
    # NB: epsilon depends on input, i.e. if images are [0, 1] then eps should be of the same order of magnitude
    #     if images are [0, 255], then eps has to be on that order of magnitude
    attack = FastGradientMethod(art_model, norm=norm, eps=eps, minimal=minimal, num_random_init=10, batch_size=256)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    # model, art_model, images, preprocessed_images, \
    # correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()
    #
    # # note: good inf norm epsilons: 1, 5, 10
    # for eps in [1, 5, 10]:
    #     images1, predictions = fgm(art_model, images, eps=eps)
    #     y_pred = np.argmax(predictions, axis=1)
    #     adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    #     display_images(adv, (4, 4))
    #
    # # note: good 1 norm epsilons: 30k, probably don't need more
    # images1, predictions = fgm(art_model, images, norm=1, eps=300000)
    # y_pred = np.argmax(predictions, axis=1)
    # adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    # display_images(adv, (4, 4))
    #
    # # note: good 2 norm epsilons: 500, 2000
    # for eps in [500, 2000]:
    #     images1, predictions = fgm(art_model, images, norm=2, eps=eps)
    #     y_pred = np.argmax(predictions, axis=1)
    #     adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    #     display_images(adv, (4, 4))

    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    x, y = get_some_imagenet_set()
    y_pred = np.argmax(art_model.predict(x), axis=1)
    x1, _ = split_correct_classification(x, y_pred, y)

    save_numpy_array()

    _, art_model2, _, _, \
    _, _, _ = setup_imagenet_model('vgg19')

    y_pred2 = np.argmax(art_model2.predict(x), axis=1)
    x2, _ = split_correct_classification(x, y_pred, y)

    same_pred = np.where(x1 == x2)

    adv_for_eps = []
    epss = [0.05, 0.1, 0.2, 0.5, 0.7, 1, 2, 3, 5, 10, 20, 30]
    for eps in epss:
        images1, predictions = fgm(art_model, images, eps=eps)
        y_pred = np.argmax(predictions, axis=1)
        adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
        adv_for_eps.append(adv)

    a = 5
    # save_images_plus_arrays(adv, subdirectory='fgm/norm_inf/eps_10', name_prefix='adv')
