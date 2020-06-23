from util import *

preload_tensorflow()

from art.attacks.evasion import FastGradientMethod


def fgm(art_model, images, eps, norm=np.inf, minimal=False):
    # NB: epsilon depends on input, i.e. if images are [0, 1] then eps should be of the same order of magnitude
    #     if images are [0, 255], then eps has to be on that order of magnitude
    attack = FastGradientMethod(art_model, norm=norm, eps=eps, minimal=minimal, num_random_init=10, batch_size=256)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


def fgm_inf():
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    # note: good inf norm epsilons: 1, 5, 10
    for eps in [2.05, 2.1, 2.2, 10, 20, 60]:
        images1, predictions = fgm(art_model, images, eps=eps)
        y_pred = np.argmax(predictions, axis=1)
        adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
        print(decode_predictions(predictions))
        # display_images(adv, (1, 3))


def fgm_1():
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    # note: good 1 norm epsilons: 30k, probably don't need more
    images1, predictions = fgm(art_model, images, norm=1, eps=300000)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
    display_images(adv, (4, 4))


def fgm_2():
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(img_range=1)

    for eps in [0.3, 0.5, 0.8, 1]:
        images1, predictions = fgm(art_model, images, norm=2, eps=eps)
        y_pred = np.argmax(predictions, axis=1)
        adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)
        try:
            display_images(adv, (4, 4))
        except:
            pass


def gen_fgm_for_resnet():
    pass


def test_resnet_fgm_on_vgg():
    pass


if __name__ == '__main__':
    setup_logging()
    fgm_inf()
    # fgm_2()
