import glob
import ntpath
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


EPOCH_COUNT = 15
FORCE_RETRAIN = False
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_untrained_model_pytorch():
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1))
            self.conv2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
            self.conv3 = nn.Conv2d(32, 64, 3, padding=(1, 1))
            self.conv4 = nn.Conv2d(64, 64, 3, padding=(1, 1))
            self.dropout25 = nn.Dropout(p=0.25)
            self.dropout50 = nn.Dropout(p=0.50)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.dropout25(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = self.dropout25(x)
            x = x.view(-1, 64 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.dropout50(x)
            x = self.fc2(x)
            return x

    net = Net()
    return net


def get_untrained_model_tf(input_shape=(32, 32, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(10)
    ])

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return model, probability_model


def setup_imagenet_model(which='resnet50v2', img_range=255,
                         classifier_activation='softmax', preprocessing_defences=None, postprocessing_defences=None):
    from art.classifiers import TensorFlowV2Classifier

    assert img_range in (1, 255)
    images, correct_labels = load_personal_images()
    images = images.astype(np.float64)

    def _setup_resnetv2_50():
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
        model = ResNet50V2(classifier_activation=classifier_activation)
        art_preprocessing = preprocessing_art_tuple('tf')
        return model, art_preprocessing, preprocess_input, decode_predictions

    def _setup_mobilenetv2():
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
        model = MobileNetV2(classifier_activation=classifier_activation)
        art_preprocessing = preprocessing_art_tuple('tf')
        return model, art_preprocessing, preprocess_input, decode_predictions

    def _setup_densenet_121():
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
        model = DenseNet121()
        art_preprocessing = preprocessing_art_tuple('torch')
        return model, art_preprocessing, preprocess_input, decode_predictions

    def _setup_vgg16():
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        # NB: VGG16 is BGR
        # NB: preprocess_input converts RGB to BGR on its own ! ! !
        #     however, art DOESN'T and we have to pass it a BGR image that has *not* been preprocessed
        #     (because it does it on it's own too)
        model = VGG16(classifier_activation=classifier_activation)
        art_preprocessing = preprocessing_art_tuple('caffe')
        return model, art_preprocessing, preprocess_input, decode_predictions

    def _setup_vgg19():
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
        # NB: VGG19 is BGR
        # NB: preprocess_input converts RGB to BGR on its own ! ! !
        #     however, art DOESN'T and we have to pass it a BGR image that has *not* been preprocessed
        #     (because it does it on it's own too)
        model = VGG19(classifier_activation=classifier_activation)
        art_preprocessing = preprocessing_art_tuple('caffe')
        return model, art_preprocessing, preprocess_input, decode_predictions

    def _setup_xception_deprecated():
        # deprecated because it's 299, it's easier to just do 224 on all of them
        from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
        model = Xception(classifier_activation=classifier_activation)
        art_preprocessing = (127.5, 127.5)
        return model, art_preprocessing, preprocess_input, decode_predictions

    lookup_setup = {
        'resnet50v2': _setup_resnetv2_50,
        'densenet121': _setup_densenet_121,
        'vgg16': _setup_vgg16,
        'vgg19': _setup_vgg19,
        'mobilenetv2': _setup_mobilenetv2,
        'xception': _setup_xception_deprecated
    }

    model, art_preprocessing, preprocess_input, decode_predictions = lookup_setup[which]()

    # NB: if we want to go back to 0-255, just remove this stuff and clip values
    clip_values = None
    if img_range == 1:
        images = images / 255
        art_preprocessing = art_preprocessing[0] / 255, art_preprocessing[1] / 255
        clip_values = (0, 1)
    elif img_range == 255:
        clip_values = (0, 255)

    art_model = TensorFlowV2Classifier(model=model, loss_object=tf.losses.SparseCategoricalCrossentropy(),
                                       nb_classes=1000, input_shape=(224, 224, 3), clip_values=clip_values,
                                       preprocessing=art_preprocessing,
                                       preprocessing_defences=preprocessing_defences,
                                       postprocessing_defences=postprocessing_defences)
    preprocessed_images = preprocess_input(np.array(images))
    return model, art_model, images, preprocessed_images, correct_labels, preprocess_input, decode_predictions


def setup_cifar10_model(epochs=EPOCH_COUNT, preload_attempt=True, save_on_preload_fail=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range

    def _train_new_model():
        model, _ = get_untrained_model_tf()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        return model

    modelname = 'tf_cifar10_epochs%d' % epochs
    path = './saved_models/' + modelname
    if FORCE_RETRAIN:
        model = _train_new_model()
        model.save(path)
    else:
        if preload_attempt:
            try:
                model = tf.keras.models.load_model(path)
            except:
                model = _train_new_model()
                if save_on_preload_fail:
                    model.save(path)
        else:
            model = _train_new_model()
            model.save(path)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    return model, probability_model, (x_train, y_train), (x_test, y_test)


def cifar10_class_id_to_text(class_ids):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    names = []
    for class_id in class_ids:
        names.append(classes[class_id])
    return names


def split_correct_classification(x, y_correct, y_pred):
    # used to split up sucessful and failed adversarial attacks
    return x[np.argwhere(y_correct != y_pred).flatten()],\
           x[np.argwhere(y_correct == y_pred).flatten()]


def display_images(images, grid_shape, figsize=None, titles=None):
    if isinstance(list(images), list):
        images = np.array(images)

    if np.issubdtype(images.dtype, np.floating) and images.max() > 1 + 1e-6:
        images = images / 255  # this is an ok assumption I think

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=grid_shape, axes_pad=0.3)
    if not titles:
        titles = [None, ] * len(images)

    for ax, im, title in zip(grid, images, titles):
        ax.imshow(im)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title:
            ax.title.set_text(title)
    plt.show()


def resize_image(image, new_size):
    scale = new_size / max(image.size)
    scaled_image = Image.new(image.mode, (new_size, new_size), (0, 0, 0))
    resized = image.resize((int(image.width * scale), int(image.height * scale)), resample=Image.LANCZOS)
    scaled_image.paste(resized, (0, 0))
    return scaled_image


def load_personal_images(size=(224, 224)):
    images = load_images('/../data/personal_images', size)
    labels = [87, 414, 319, 22, 582, 366, 338, 607, 611, 508, 355, 299, 217, 770, 281, 894]
    return images, labels


def load_images(directory_path, size=None, crop=False):
    directory_path = ROOT_DIR + directory_path
    filelist = glob.glob(directory_path + '/*.jpg')
    images = []
    for f in filelist:
        image = Image.open(f)
        if crop:
            scaled_image = ImageOps.fit(image, size, Image.ANTIALIAS, bleed=0.1)
        else:
            # scaled_image = resize_image(image, size[0])
            scaled_image = image.resize(size, resample=Image.LANCZOS)
        images.append(np.array(scaled_image))
    return np.array(images)


def preprocessing_art_tuple(mode, shape=(224, 224, 3)):
    assert mode in ('tf', 'torch', 'caffe')

    if mode == 'tf':
        return 127.5, 127.5

    if mode == 'torch':
        preprocessing_mean = np.zeros(shape)
        preprocessing_mean[..., 0].fill(123.675)
        preprocessing_mean[..., 1].fill(116.28)
        preprocessing_mean[..., 2].fill(103.53)

        preprocessing_std = np.zeros(shape)
        preprocessing_std[..., 0].fill(0.229 * 255)
        preprocessing_std[..., 1].fill(0.224 * 255)
        preprocessing_std[..., 2].fill(0.225 * 255)

        return preprocessing_mean, preprocessing_std

    if mode == 'caffe':
        # NB: caffe is BGR, not RGB
        preprocessing_mean = np.zeros(shape)
        preprocessing_mean[..., 0].fill(103.939)
        preprocessing_mean[..., 1].fill(116.779)
        preprocessing_mean[..., 2].fill(123.68)

        return preprocessing_mean, 1


def save_numpy_array_as_image(np_array, name, directory='/results/images/'):
    from PIL import Image

    if np.issubdtype(np_array.dtype, np.floating) and np_array.max() <= 1 + 1e-6:
        np_array = np_array * 255
    np_array_int = np_array.astype(np.uint8)

    img = Image.fromarray(np_array_int, mode="RGB")
    fullname = '%s%s.jpg' % (directory, name)
    img.save(fullname)


def save_images_plus_arrays(np_arrays, directory='/results/images', subdirectory=None, names=None, name_prefix=None):
    if not names:
        names = np.arange(len(np_arrays))

    if name_prefix:
        name_prefix += '_'
    else:
        name_prefix = ''

    if subdirectory:
        directory = directory + '/' + subdirectory
    directory = ROOT_DIR + directory + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for arr, name in zip(np_arrays, names):
        whole_name = '%s%s' % (name_prefix, name)
        save_numpy_array(arr, whole_name, directory)
        save_numpy_array_as_image(arr, whole_name, directory)


def save_numpy_array(np_array, name, directory='results'):
    path = os.path.join(directory, name)
    np.save(path, np_array)


def load_numpy_array(name, directory='results'):
    path = os.path.join(directory, name + '.npy')
    return np.load(path)


def preload_tensorflow():
    # nobody should ever ask me what this does. ever.
    import tensorflow as tf
    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)
    y = tf.keras.layers.Conv2D(
        2, 3, activation='relu', input_shape=input_shape)(x)
    print('loaded tf')


def norm_between(x, y, norm=1):
    assert norm in (1, 2, np.inf)
    if norm == np.inf:
        return np.abs(x - y).max()
    if norm == 2:
        return np.sqrt(np.sum((x - y) ** 2))
    if norm == 1:
        return np.sum(np.abs(x - y))


def power_spectrum(image):
    from scipy import fftpack
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    p_spectrums = []
    p_avgs = []
    for color_channel in (r, g, b):
        f1 = fftpack.fft2(color_channel)
        f2 = fftpack.fftshift(f1)
        p_spectrum = np.abs(f2) ** 2
        p_avg = azimuthal_avg(p_spectrum)
        p_avgs.append(p_avg)
        p_spectrums.append(p_spectrum)

    t = [azimuthal_avg(np.log(z + 1e-8) / 26) for z in p_spectrums]
    return t


def azimuthal_avg(img):
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    r = np.sqrt((x - 112) ** 2 + (y - 112) ** 2)
    calc_mean = lambda some_r: img[(r >= some_r - 0.5) & (r < some_r + 0.5)].mean()
    radii = np.linspace(1, 112, num=112)  # hardcoded I guess
    azimuthal = np.vectorize(calc_mean)(radii)  # vectorized makes it speedy
    return azimuthal


def setup_logging():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_some_imagenet_set():
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    val = datasets.ImageFolder(
        ROOT_DIR + '/../data/imagenet_validation',
        transforms.Compose([
            transforms.Resize(256, Image.BICUBIC),
            transforms.CenterCrop(224),
        ]))

    x = np.array([np.array(img[0]) for img in val]).astype(np.float32)
    y = np.array(val.targets)
    return x, y


if __name__ == '__main__':
    setup_cifar10_model(69)
    # preload_tensorflow()
    # images = load_images('../data/personal_images', (224, 224))[1:]
    # images = images.astype(np.float32)
    #
    # all_pwrs = []
    # for img in images:
    #     pwrs = power_spectrum(img)
    #     pwr = np.sum(pwrs, axis=0) / 3
    #     all_pwrs.append(pwr)
    # all_pwrs = np.sum(np.array(all_pwrs), axis=0) / len(images)
    #
    # all_pwrs_noisy = []
    # noisy_images = []
    # for img in images:
    #     img_noisy = img + (np.random.random((224, 224, 3)) - 0.5) * 0.011113533
    #     img_noisy = np.clip(img_noisy, 0, 255)
    #     noisy_images.append(img_noisy)
    #     pwrs = power_spectrum(img_noisy)
    #     pwr = np.sum(pwrs, axis=0) / 3
    #     all_pwrs_noisy.append(pwr)
    # all_pwrs_noisy = np.sum(np.array(all_pwrs_noisy), axis=0) / len(images)
    #
    # from art.attacks.evasion import FastGradientMethod
    # model, art_model, _images, preprocessed_images, preprocess_input, decode_predictions = setup_imagenet_model()
    #
    # norms = [np.inf, 1, 2]
    # epsilons = [0.5, 1, 2, 5, 10, 20]
    # attack = FastGradientMethod(art_model, eps=80, minimal=True)
    # adversarial_images = attack.generate(images)
    # adversarial_predictions = decode_predictions(art_model.predict(adversarial_images))
    # a = 5
    #
    # all_pwrs_adv = []
    # for img in adversarial_images:
    #     pwrs = power_spectrum(img)
    #     pwr = np.sum(pwrs, axis=0) / 3
    #     all_pwrs_adv.append(pwr)
    # all_pwrs_adv = np.sum(np.array(all_pwrs_adv), axis=0) / len(images)
    #
    # z = 0
