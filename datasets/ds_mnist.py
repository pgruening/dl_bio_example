import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from config import DATA_FOLDER


def get_dataloader(
    is_train=True, indeces=None, batch_size=32, num_workers=0,
        data_path=DATA_FOLDER):
    # https://discuss.pytorch.org/t/train-on-a-fraction-of-the-data-set/16743/6
    dataset = get_dataset(is_train, data_path)

    if indeces is None:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            sampler=SubsetRandomSampler(indeces)
        )
    return data_loader


def get_dataset(is_train=True, data_path=DATA_FOLDER):
    dataset = torchvision.datasets.MNIST(
        data_path,
        train=is_train,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )

    return dataset


if __name__ == "__main__":
    from DLBio.pytorch_helpers import cuda_to_numpy
    import matplotlib.pyplot as plt
    dataset = get_dataset()
    for x, y in dataset:
        x -= x.min()
        x /= x.max()
        plt.imshow((255. * cuda_to_numpy(x)).astype('uint8'))
        plt.show()