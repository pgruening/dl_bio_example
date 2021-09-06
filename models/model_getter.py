"""
The model_getter manages to initialization of a PyTorch neural network.

A model that process images usually expect a Tensor of shape
(b, d, h, w) as input. 'b' being the mini-batch size: how many images per 
batch, 'd' the number of input channels –usually three for red, green, and 
blue– and the height 'h' and width 'w' of the image. Note that each image in 
the mini-batch needs to have the same height and width, and number of channels.
To achieve the equal shapes of all images, they are usually cropped or resized
before combining them to a mini-batch.

The function get_model expects a string 'model_type' to define which model 
needs to be loaded. With 'input_dim', the number of input channels is defined, 
usually three for RGB or one for grayscale images. 'output_dim' defines the 
number of output classes. For example, the cifar-10 dataset has images of ten 
different classes (cat, dog, plane, ...); thus, you must set the output 
dimension to ten.

You can use kwargs (keyword-arguments) to pass configuration values that are 
specific to a model. For example, pre_trained is used to load a model with 
already trained weights, e.g., on ImageNet.
In 'run_training.py', the model kwargs are defined by the argument 'model_kw'. 
Here, DLBio's 'kwargs_translator' is used: all values are lists and need to be 
typecasted if they are floats or ints.
"""
from torchvision.models import resnet18, vgg16_bn, alexnet
import torch.nn as nn

from DLBio.kwargs_translator import get_kwargs

from models.costum_archs import CustomNet


def get_model(model_type: str, input_dim: int, output_dim: int, device, **kwargs):
    # add pre_trained
    pretrained = kwargs.get('pretrained', [False])[0]
    if model_type == 'resnet18':
        assert input_dim == 3
        model = resnet18(pretrained=pretrained)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, output_dim)
        return model.to(device)

    elif model_type == 'vgg16':
        assert kwargs['crop_size'] == 224
        assert input_dim == 3
        model = vgg16_bn(pretrained=pretrained)
        in_dim = model.classifier[-1].in_features
        model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
        return model.to(device)

    elif model_type == 'alexnet':
        assert kwargs['crop_size'] == 256
        assert input_dim == 3
        model = alexnet(pretrained=pretrained)
        in_dim = model.classifier[-1].in_features
        model.model.classifier[-1] = nn.Linear(in_dim, output_dim)
        return model.to(device)

    elif model_type == 'custom_net':
        model = CustomNet(input_dim, output_dim, **kwargs)

        return model.to(device)


def get_weights(model_name):
    """
    get pretrained weights or download them from google drive first 

    usage of drive links to download files:
            share the corrsponding document such that anyone with the link can see it
            copy the link and use only the file id for
                    download_file_from_google_drive from torchvision.datasets.utils
                    https://github.com/pytorch/vision/blob/1b7c0f54e2913e159394a19ac5e50daa69c142c7/torchvision/datasets/utils.py#L169

            example:
                    link: https://drive.google.com/file/d/14M3uC29aAx2AMeCeidLQjqjkVpGqnb6k/view?usp=sharing
                    with id = 14M3uC29aAx2AMeCeidLQjqjkVpGqnb6k


    Parameters
    ----------
    model_name : str
            name of file_id saved in config.WEIGHT_IDS

    Returns
    -------
    state_dict
            state_dict containing pretrained weights
    """
    from os.path import join, exists
    import config
    import torch
    from torchvision.datasets.utils import download_file_from_google_drive

    pretrained_path = join(config.DATA_FOLDER, 'pretrained_models')
    assert model_name in config.WEIGHT_IDS, f'no weights for model: {model_name}'

    file_name = f'{model_name}.pt'
    file_dest = join(pretrained_path, file_name)

    if not exists(file_dest):
        id = config.WEIGHT_IDS[model_name]
        download_file_from_google_drive(
            file_id=id, root=pretrained_path, filename=file_name)
        print(f'weights downloaded & saved at: {file_dest}')

    return torch.load(file_dest)


def load_model_from_opt(options, device):
    """wrapper for get_model to work with only options and device as parameters
    needed for dlbio's 'load_model_with_opt'

    Returns
    -------
    pytorch model
            the pytorch model from get_model
    """
    mt = options.model_type
    in_dim = options.in_dim
    out_dim = options.out_dim
    model_kw = get_kwargs(options.model_kw)

    if hasattr(options, "model_kw"):
        kw = options.model_kw
        model_kw = get_kwargs(kw)
    else:
        model_kw = dict()

    return get_model(mt, in_dim, out_dim, device, False, **model_kw)
