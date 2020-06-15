import torch
import torch.hub


def resnet_18():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18')
    model.fc = torch.nn.Linear(512, 10)
    return model
