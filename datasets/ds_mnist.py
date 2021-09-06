import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from config import DATA_FOLDER


def get_dataloader(
    is_train=True, batch_size=32, num_workers=0,
        data_path=DATA_FOLDER, **kwargs):

    convert_to_rgb = kwargs.get('convert_to_rgb', [False])[0]
    dataset = get_dataset(is_train, data_path, convert_to_rgb=convert_to_rgb)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers
    )
    return data_loader


def get_dataset(is_train=True, data_path=DATA_FOLDER, convert_to_rgb=False):
    if convert_to_rgb:
        aug = [torchvision.transforms.Lambda(lambda x: x.convert('RGB'))]
    else:
        aug = []

    aug += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]
    aug = torchvision.transforms.Compose(aug)

    dataset = torchvision.datasets.MNIST(
        data_path,
        train=is_train,
        download=True,
        transform=aug
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
