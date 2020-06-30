from util import *

preload_tensorflow()

from art.attacks.evasion import CarliniLInfMethod, CarliniL2Method


def cw_l2(art_model, images, max_iter=5, confidence=0.0, target=None, learning_rate=0.02, initial_const=0.01):
    targeted = False
    if target:
        targeted = True
    attack = CarliniL2Method(art_model, max_iter=max_iter, learning_rate=learning_rate,
                             confidence=confidence, targeted=targeted, initial_const=initial_const, binary_search_steps=10)
    adversarial_images = attack.generate(images, y=target)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


def cw_linf(art_model, images, eps=5, confidence=0.0, max_iter=120, learning_rate=0.5, target=None):
    targeted = False
    if target:
        targeted = True
    attack = CarliniLInfMethod(art_model, max_iter=max_iter,
                               learning_rate=learning_rate, eps=eps, confidence=confidence, targeted=targeted)
    adversarial_images = attack.generate(images, y=target)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


def cw_false_negative(art_model, n=1, confidence=0.95, max_iter=20, learning_rate=0.5, *args):
    img = np.array([np.random.random((224, 224, 3)) * 255 for _ in range(n)])
    attack = CarliniL2Method(art_model, confidence=confidence, max_iter=max_iter, learning_rate=learning_rate)
    adversarial_images = attack.generate(img)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    setup_logging()
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(classifier_activation=None)

    _, art_model2, _, _, _, _, _ = setup_imagenet_model()  # just to get prob model
    # images = np.array([images[6], images[7], images[9], images[10]])  # good mix of images

    advs1 = []
    for img in images:
        adv, _ = cw_linf(art_model, np.array([img]), max_iter=50, eps=6)
        adv = adv[0]
        advs1.append(adv)
    advs1 = np.array(advs1)
    # diff1 = ((advs1 - images) - (advs1 - images).min())
    # diff1 = diff1 / diff1.max()

    advs2 = []
    for img in images:
        adv, _ = cw_l2(art_model, np.array([img]), confidence=0, max_iter=10, initial_const=0.01, learning_rate=0.2)
        adv = adv[0]
        advs2.append(adv)
    advs2 = np.array(advs2)
    # diff2 = ((advs2 - images) - (advs2 - images).min())
    # diff2 = diff2 / diff2.max()
