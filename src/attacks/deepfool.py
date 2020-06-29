from util import *

preload_tensorflow()

from art.attacks.evasion import DeepFool


def deepfool(art_model, images, eps=1e-6, max_iter=100):
    attack = DeepFool(art_model, epsilon=eps, max_iter=max_iter, nb_grads=10, batch_size=10)
    adversarial_images = attack.generate(images)
    adversarial_predictions = art_model.predict(adversarial_images)
    return adversarial_images, adversarial_predictions


def deepfool_cifar10():
    from art.classifiers import TensorFlowV2Classifier
    from art.utils import to_categorical
    from art.attacks.evasion import DeepFool
    from art.defences.preprocessor import ThermometerEncoding

    model1, probability_model1, (x_train, y_train), (x_test, y_test) = setup_cifar10_model(epochs=5)
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    art_model1 = TensorFlowV2Classifier(model1, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1),
                                       loss_object=model1.loss)
    attack = DeepFool(art_model1)
    adversarial_images = attack.generate(x_test[:10])

    # # # #

    model2, probability_model2 = get_untrained_model_tf((32, 32, 30))
    # model2 = tf.keras.models.load_model('./saved_models/thermometer_cifar10_5')

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    defence = ThermometerEncoding(clip_values=(0, 1), num_space=10)
    art_model2 = TensorFlowV2Classifier(model2, nb_classes=10, input_shape=(32, 32, 30), clip_values=(0, 1),
                                        preprocessing_defences=defence, train_step=train_step,
                                        loss_object=loss_object)

    import time
    t1 = time.time()
    art_model2.fit(x_train, to_categorical(y_train, 10), nb_epochs=30)
    t2 = time.time()
    print('time %f' % (t2 - t1))

    modelname = 'thermometer_cifar10_30_postfix_10space'
    path = './saved_models/' + modelname
    model2.save(path)

    attack2 = DeepFool(art_model2)
    adversarial_images2 = attack.generate(x_test[:10])
    a = 5


if __name__ == '__main__':
    setup_logging()
    model, art_model, images, preprocessed_images, \
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(classifier_activation=None, img_range=1)

    images = np.array([images[0], images[11]])
    correct_labels = np.array([correct_labels[0], correct_labels[11]])

    for img in images:
        adv, prediction = deepfool(art_model, np.array([img]), eps=1e-6, max_iter=50)
        y_pred = np.argmax(prediction, axis=1)
        print(decode_predictions(tf.nn.softmax(prediction).numpy()))
        adv = adv[0]
        diff = adv - img
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        display_images([adv, diff], (1, 2))
