from util import *
preload_tensorflow()


def jpeg_compression(quality=50):
    from art.attacks.evasion import FastGradientMethod
    from art.defences.preprocessor import JpegCompression

    model, art_model1, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()

    y_pred = np.argmax(art_model1.predict(images), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('no jpeg model on clean %.4f' % acc)

    attack = FastGradientMethod(art_model1, eps=1)
    f1 = attack.generate(images)
    y_pred = np.argmax(art_model1.predict(f1), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('no jpeg model on fgsm 1 %.4f' % acc)

    attack = FastGradientMethod(art_model1, eps=5)
    f2 = attack.generate(images)
    y_pred = np.argmax(art_model1.predict(f2), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('no jpeg model on fgsm 5 %.4f' % acc)

    attack = FastGradientMethod(art_model1, eps=10)
    f3 = attack.generate(images)
    y_pred = np.argmax(art_model1.predict(f3), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('no jpeg model on fgsm 10 %.4f' % acc)

    defense = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=quality)
    clean, _ = defense(images)
    f1, _ = defense(f1)
    f2, _ = defense(f2)
    f3, _ = defense(f3)

    y_pred = np.argmax(art_model1.predict(clean), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('jpeg model on clean %.4f' % acc)
    y_pred = np.argmax(art_model1.predict(f1), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('jpeg model on fgsm 1 %.4f' % acc)
    y_pred = np.argmax(art_model1.predict(f2), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('jpeg model on fgsm 5 %.4f' % acc)
    y_pred = np.argmax(art_model1.predict(f3), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('jpeg model on fgsm 10 %.4f' % acc)

    from attacks.deepfool import deepfool
    defense = JpegCompression(clip_values=(0, 1), apply_predict=True, quality=quality)
    model, art_model1, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = \
        setup_imagenet_model(img_range=1, classifier_activation=None, preprocessing_defences=defense)

    all_adv = []
    for img in images:
        adv, prediction = deepfool(art_model1, np.array([img]), eps=1e-6, max_iter=50)
        adv = adv[0]
        all_adv.append(adv)
    all_adv = np.array(all_adv)

    y_pred = np.argmax(art_model1.predict(all_adv), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('no jpeg model on deepfool %.4f' % acc)

    all_adv_jpg, _ = defense(all_adv * 255)
    all_adv_jpg = all_adv_jpg / 255

    y_pred = np.argmax(art_model1.predict(all_adv_jpg), axis=1)
    acc = np.sum(y_pred == correct_labels) / len(correct_labels)
    print('jpeg model on deepfool %.4f' % acc)


def jpeg_visualize():
    from art.defences.preprocessor import JpegCompression

    images, _ = load_personal_images((224, 224))
    # img = np.array([images[9]])
    orig_img = images[9] / 1.0
    d1 = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=5)
    d2 = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=25)
    d3 = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=50)
    d4 = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=90)

    compressed_imgs = []
    for d in [d1, d2, d3, d4]:
        img, _ = d(np.array([orig_img]))
        img = img[0]
        compressed_imgs.append(img)
    display_images(compressed_imgs, (1, 4))


if __name__ == '__main__':
    setup_logging()
    jpeg_compression(quality=25)
    # jpeg_visualize()
