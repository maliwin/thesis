from util import *

preload_tensorflow()
setup_logging()

from art.attacks.evasion import DeepFool


def deepfool(art_model, images, eps=1e-6, max_iter=100):
    attack = DeepFool(art_model, epsilon=eps, max_iter=max_iter)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


if __name__ == '__main__':
    # TODO: just switch to cifar10, imagenet is too slow for deepfool
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    # note:
    images1, predictions = deepfool(art_model, images, eps=5, max_iter=1)
    y_pred = np.argmax(predictions, axis=1)
    adv, not_adv = split_correct_classification(images1, y_pred, correct_labels)

    a = 5
    # save_images_plus_arrays(adv, subdirectory='fgm/norm_inf/eps_10', name_prefix='adv')
