# attacks
# defenses
# attacks
# defenses

# images, predictions, values
from util import *
preload_tensorflow()
from adversarial_attacks import *


def save_adversarial_images(images, name):
    # name should be [model]_[attack]_[meta_info]
    save_numpy_array(images, name)


def generate_boundary_results():
    def _single_image_targeted_boundary_attack():
        x_advs, prediction_history = boundary_attack(init_image_idxs=[10], target_image_idxs=[2],
                                                     targeted=True, which_model='mobilenetv2', iter_count=1)
        # name = 'resnet50v2_boundary_targeted-single-image'
        name = 'null'
        # save_adversarial_images(x_advs, name)
        save_numpy_array(x_advs, name)
        loaded = load_numpy_array(name)
        assert np.all(x_advs == loaded)

    def _multi_image_targeted_boundary_attack():
        print('multi image attack')
        # x_advs, prediction_history = boundary_attack(init_image_idxs=[10, 7, 0],
        #                                              target_image_idxs=[2, 2, 2], targeted=True,
        #                                              iter_step=400)
        name = 'resnet50v2_boundary_targeted-multi-image-long'
        # save_adversarial_images(x_advs, name)
        # save_numpy_arrays(x_advs, name)
        loaded = load_numpy_array(name)
        a = 5
        # assert np.all(x_advs == loaded)

    def _single_image_untargeted_boundary_attack():
        x_advs, prediction_history = boundary_attack(init_image_idxs=[12], target_image_idxs=[5], targeted=True)
        name = 'resnet50v2_boundary_targeted-single-image'
        # save_adversarial_images(x_advs, name)
        save_numpy_array(x_advs, name)
        loaded = load_numpy_array(name)
        assert np.all(x_advs == loaded)

    # _single_image_targeted_boundary_attack()
    _multi_image_targeted_boundary_attack()


def display_boundary_results():
    def _single_image_targeted_boundary_attack():
        pass

    def _multi_image_targeted_boundary_attack():
        name = 'resnet50v2_boundary_targeted-multi-image'
        loaded = load_numpy_array(name)
        loaded = loaded / 255
        first, second = loaded[:10], loaded[10:]
        first = first.reshape(30, 224, 224, 3)
        second = second.reshape(30, 224, 224, 3)
        display_images(first, (10, 3))
        display_images(second, (10, 3))
        a = 5

    def _single_image_untargeted_boundary_attack():
        pass

    # _single_image_targeted_boundary_attack()
    _multi_image_targeted_boundary_attack()


if __name__ == '__main__':
    generate_boundary_results()
    # display_boundary_results()
