def jpeg_compression():
    from art.attacks.evasion import FastGradientMethod
    from art.defences.preprocessor import JpegCompression

    model, art_model, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model()
    attack = FastGradientMethod(art_model, eps=5)
    no_jpeg = attack.generate(images)

    defense = JpegCompression(clip_values=(0, 255), apply_predict=True, quality=50)
    model, art_model, images, preprocessed_images,\
    correct_labels, preprocess_input, decode_predictions = setup_imagenet_model(preprocessing_defences=[defense])
    attack = FastGradientMethod(art_model, eps=5)
    jpeg = attack.generate(images)
    a = 5
