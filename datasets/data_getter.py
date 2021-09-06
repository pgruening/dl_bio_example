"""
# data_getter
The data getter manages the initialization of different torch DataLoaders.
A dataloader is essentially an Iterable that can be called in a for-loop.

A typical training step could for example look like this:

data_loaders = data_getter.get_data_loaders(...)
for sample in data_loader['train']:
    image, label = sample[0], sample[1]
    prediction = model(image)
    loss = loss_function(prediction, label)
    ...


A dataloader contains an object of class Dataset that handles the loading and augmentation process. 'ds_natural_images' gives an example for a custom 
dataset.

'get_data_loaders' expects a string 'dataset' that identifies which dataset is to be used (e.g., mnist, cifar-10, ...). 'batch_size' denotes how many
samples (here mainly images) are combined to a mini-batch. A typical PyTorch
minibatch tensor of images has the dimension:
(batch_size, 3, height of image, width of image)
3 is the dimension of the three image channels red, green, and blue.
In 'run_training.py', batch_size is defined by the argument 'bs'.

'num_workers' defines how many processes load data in parallel. Using more than 
one worker can, in specific cases, speed up the dataset loading process and 
, thus, the entire training. If you want to debug your code, num_workers needs
to be set to 0.
In 'run_training.py', num_workers is defined by the argument 'nw'.

You can use kwargs (in 'run_training.py' the system argument 'ds_kwargs') to
pass configuration values that are very specific to a dataset.
kwargs is a dictionary of keyword-value pairs. EACH VALUE IS A LIST, even if it 
only contains a single element. Furthermore, you need to take care of each 
value's type. For example,

split_index = int(kwargs['split_index'][0])

contains a list with a string. To get the actual number, you'll need to typecast
it to an int.
For more information, see DLBio's 'kwargs_translator'.

To add a new dataset, you'll need to create a new file 'ds_[dataset_name].py' 
in the 'data' folder. You'll need to create a class that inherits Dataset and
implements '__getitem__' and '__len__'. Furthermore, you'll need to define the
function 'get_dataloader'. Finally, you'll need to append an elif case to this
module's function 'get_data_loaders' that calls 'get_dataloader' and returns
a dictionary containing the keys 'train', 'val', and 'test'. If there is no
'val' or 'test' dataloader available, set these values to None.
'ds_natural_images.py' is an example of how to write a custom dataset.

"""
from . import ds_natural_images
from . import ds_cifar10
from . import ds_mnist


def get_data_loaders(dataset: str, batch_size: int, num_workers: int, **kwargs) -> dict:
    if dataset == 'nat_im':
        split_index = int(kwargs['split_index'][0])
        return {'train': ds_natural_images.get_dataloader(
            True, batch_size, split_index, num_workers=num_workers),
            'val': ds_natural_images.get_dataloader(
            False, batch_size, split_index, num_workers=num_workers),
            'test': None
        }

    elif dataset == 'cifar_10':
        return {
            'train': ds_cifar10.get_data_loader(
                is_train=True, batch_size=batch_size,
                num_workers=num_workers
            ),
            'val': ds_cifar10.get_data_loader(
                is_train=False, batch_size=batch_size,
                num_workers=num_workers
            ),
            'test': None
        }

    elif dataset == 'mnist':
        return{
            'train': ds_mnist.get_dataloader(
                True, batch_size=batch_size, num_workers=num_workers),
            'val': ds_mnist.get_dataloader(
                False, batch_size=batch_size, num_workers=num_workers),
            'test': None
        }
    else:
        raise ValueError(f'Unkown dataset: {dataset}')
