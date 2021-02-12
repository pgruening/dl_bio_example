from . import ds_natural_images
from . import ds_mnist


def get_data_loaders(dataset, batch_size, **kwargs):
    if dataset == 'nat_im':
        return {'train': ds_natural_images.get_dataloader(
            True, batch_size, kwargs['split_index']),
            'val': ds_natural_images.get_dataloader(
            False, batch_size, kwargs['split_index']),
            'test': None
        }
    elif dataset == 'mnist':
        return{
            'train': ds_mnist.get_dataloader(
                True, batch_size=batch_size),
            'val': ds_mnist.get_dataloader(
                False, batch_size=batch_size),
            'test':None
        }
    else:
         raise ValueError(f'Unkown dataset: {dataset}')
