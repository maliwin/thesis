from util import *
preload_tensorflow()

from attacks.fgm import fgm
from attacks.deepfool import deepfool


def make_resnet50_fgsm():
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(classifier_activation=None, img_range=1)

    print(np.sum(np.argmax(art_model.predict(images), axis=1) == correct_labels) / len(correct_labels))
    adv, _ = fgm(art_model, images, eps=0.05)
    print(np.sum(np.argmax(art_model.predict(adv), axis=1) == correct_labels) / len(correct_labels))
    save_numpy_array(adv, 'resnet50_fgsm_adv', '.')

    adv2 = []
    for img in images:
        adv_img, _ = deepfool(art_model, np.array([img]), max_iter=30)
        adv2.append(adv_img[0])
    adv2 = np.array(adv2)
    print(np.sum(np.argmax(art_model.predict(adv2), axis=1) == correct_labels) / len(correct_labels))
    save_numpy_array(adv2, 'resnet50_deepfool_adv', '.')


def test_densenet_transfer():
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model('densenet121', img_range=1)

    print(np.sum(np.argmax(art_model.predict(images), axis=1) == correct_labels) / len(correct_labels))
    adv_imgs = load_numpy_array('resnet50_fgsm_adv', '.')
    print(np.sum(np.argmax(art_model.predict(adv_imgs), axis=1) == correct_labels) / len(correct_labels))
    adv_imgs = load_numpy_array('resnet50_deepfool_adv', '.')
    print(np.sum(np.argmax(art_model.predict(adv_imgs), axis=1) == correct_labels) / len(correct_labels))


if __name__ == '__main__':
    setup_logging()
    make_resnet50_fgsm()
    # test_densenet_transfer()
