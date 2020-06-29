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
    # model, art_model, images, preprocessed_images, \
    # correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()
    from tensorflow.keras.utils import to_categorical
    from art.classifiers import TensorFlowV2Classifier
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    y_train, y_test = y_train.flatten(), y_test.flatten()

    model, prob_model = get_untrained_model_tf()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, to_categorical(y_train, 10), epochs=10,
              validation_data=(x_test, to_categorical(y_test, 10)))

    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3),
                                        clip_values=(0, 1),
                                       loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    for eps in [2 / 255, 5 / 255, 10 / 255]:
        images1, predictions = fgm(art_model, x_test, eps=eps)
        y_pred = np.argmax(predictions, axis=1)


    x_train, x_test = x_train.astype(np.float32) * 255, x_test.astype(np.float32) * 255
    model, prob_model = get_untrained_model_tf()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, to_categorical(y_train, 10), epochs=20,
              validation_data=(x_test, to_categorical(y_test, 10)))

    art_model = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(32, 32, 3),
                                        clip_values=(0, 255),
                                       loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    for eps in [2, 5, 10]:
        images1, predictions = fgm(art_model, x_test, eps=eps)
        y_pred = np.argmax(predictions, axis=1)



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
