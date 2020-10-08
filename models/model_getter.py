from torchvision.models import resnet18, vgg16_bn, alexnet
import torch.nn as nn


def get_model(model_type, input_dim, output_dim, device, **kwargs):
    if model_type == 'resnet18':
        assert input_dim == 3
        model = resnet18(pretrained=True)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, output_dim)
        return model.to(device)

    elif model_type == 'vgg16':
        assert kwargs['crop_size'] == 224
        assert input_dim == 3
        model = vgg16_bn(pretrained=True)
        in_dim = model.classifier[-1].in_features
        model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
        return model.to(device)

    elif model_type == 'alexnet':
        assert kwargs['crop_size'] == 256
        assert input_dim == 3
        model = alexnet(pretrained=True)
        in_dim = model.classifier[-1].in_features
        model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
        return model.to(device)

    elif model_type == 'my_model':
        pass
