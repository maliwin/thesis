from util import *
# preload_tensorflow()

setup_logging()


def train_pytorch():
    import torchvision
    import torchvision.transforms as transforms

    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.optim as optim

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    model = get_untrained_model_pytorch()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    param_count = 0
    for param in model.named_parameters():
        name, params = param
        param_count += len(params)
    print('Param count: %d' % param_count)

    for epoch in range(25):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        assert loss
        print("epoch %d loss: %.3f" % (epoch, loss.item()))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    torch.save(model.state_dict(), './transferability/pytorch.pth')


def train_tf():
    preload_tensorflow()
    import tensorflow as tf

    model, _ = get_untrained_model_tf()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255  # (0, 1) range
    model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))
    # model.save('./transferability/tf')


def get_tf_model():
    return tf.keras.models.load_model('tf')


def get_pytorch_model():
    import torch
    net = get_untrained_model_pytorch()
    net.load_state_dict(torch.load('./transferability/pytorch.pth'))
    return net


def transferability():
    import torch
    def pytorch_generate_adv():
        from art.attacks.evasion import DeepFool, FastGradientMethod
        from art.classifiers import PyTorchClassifier
        from art.utils import load_cifar10

        (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
        art_model_pytorch = PyTorchClassifier(get_pytorch_model(), loss=torch.nn.CrossEntropyLoss(),
                                              input_shape=(32, 32, 3), nb_classes=10, clip_values=(min_, max_))
        attack = FastGradientMethod(art_model_pytorch, norm=2)
        attack_set = x_test[:25].swapaxes(1, 3).astype(np.float32)
        x_adv = attack.generate(attack_set)
        x_adv = x_adv.swapaxes(1, 3)
        save_numpy_array(x_adv, 'x_adv', './transferability')

    pytorch_generate_adv()


if __name__ == '__main__':
    train_tf()
    # train_pytorch()
    # transferability()
