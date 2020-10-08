import glob
import json
import random
import re
from os.path import join

import cv2
import torch
import torchvision
from DLBio.helpers import check_mkdir, search_in_all_subfolders
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import config

CLASSES = [
    'airplane',
    'car',
    'cat',
    'dog',
    'flower',
    'fruit',
    'motorbike',
    'person'
]


def get_dataloader(is_train, batch_size, split_index=0):
    split = _find_split_file(int(split_index))
    if is_train:
        dataset = NatImDataset(split['train'], _get_data_aug(is_train))
    else:
        dataset = NatImDataset(split['test'], _get_data_aug(is_train))

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def _get_data_aug(is_train, crop_size=224):
    if is_train:
        aug = [
            torchvision.transforms.ToPILImage('RGB'),
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True),
            torchvision.transforms.Resize(
                (crop_size, crop_size), interpolation=2),
            torchvision.transforms.ToTensor(),
            # using imagenet normalization
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
    else:
        aug = [
            torchvision.transforms.ToPILImage('RGB'),
            #torchvision.transforms.Pad(crop_size // 2),
            # torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.Resize(
                (crop_size, crop_size), interpolation=2),
            torchvision.transforms.ToTensor(),
            # using imagenet normalization
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]

    return torchvision.transforms.Compose(aug)


class NatImDataset(Dataset):
    def __init__(self, paths, augmentation):
        def load_image(x):
            return cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)

        self.images = [load_image(x) for x in paths]
        self.labels = [_get_class(x) for x in paths]

        self.aug = augmentation

    def __getitem__(self, index):
        x = self.images[index]
        x = self.aug(x)

        y = torch.tensor([self.labels[index]]).long()

        return x, y

    def __len__(self):
        return len(self.labels)


def _find_split_file(index):
    def get_index(x):
        rgx = r'.*\/(\d+).json'
        return int(re.match(rgx, x).group(1))

    for file in glob.glob(join(config.NAT_IM_BASE, 'splits', '*.json')):
        if get_index(file) == index:
            with open(file, 'r') as file:
                split = json.load(file)
                return split

    raise ValueError(f'Could not find split{index}')


def _get_class(x):
    rgx = r'.*\/(.*)_\d\d\d\d.jpg'
    class_name = re.match(rgx, x).group(1)
    return CLASSES.index(class_name)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def create_splits(num_splits=5, split_perc=.8):
    images_ = _get_images_sorted_by_class()

    for i in range(num_splits):
        train_images = []
        test_images = []

        # for each class get split_perc % for training
        for tmp in images_.values():
            n = len(tmp)
            n_train = int(split_perc * n)

            # grab images without replacement
            tmp_train = random.sample(tmp, n_train)
            tmp_test = list(set(tmp) - set(tmp_train))

            train_images += tmp_train
            test_images += tmp_test

        out_path = join(config.NAT_IM_BASE, 'splits', f'{i}.json')
        check_mkdir(out_path)
        with open(out_path, 'w') as file:
            json.dump({
                'train': train_images,
                'test': test_images
            }, file)


def _get_images_sorted_by_class():
    all_images = search_in_all_subfolders(
        r'(.*)_(\d\d\d\d).jpg', config.NAT_IM_BASE)
    images_by_class = dict()
    for x in all_images:
        index = _get_class(x)
        if index not in images_by_class.keys():
            images_by_class[index] = [x]
        else:
            images_by_class[index].append(x)
    return images_by_class

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _test_splits():
    all_images = search_in_all_subfolders(
        r'(.*)_(\d\d\d\d).jpg', config.NAT_IM_BASE)

    for file in glob.glob(join(config.NAT_IM_BASE, 'splits', '*.json')):
        with open(file, 'r') as file:
            split = json.load(file)

        assert set(split['train'] + split['test']) == set(all_images)
        assert not set(split['train']).intersection(set(split['test']))

    print('Test succeeded.')


def _debug_dataset():
    import matplotlib.pyplot as plt
    from DLBio.pytorch_helpers import cuda_to_numpy
    from DLBio.helpers import to_uint8_image

    data_loader = get_dataloader(True, 16, split_index=0)
    for x, y in data_loader:
        print(y)

        for b in range(x.shape[0]):
            tmp = cuda_to_numpy(x[b, ...])
            tmp = to_uint8_image(tmp)

            plt.imshow(tmp)
            plt.title(int(y[b]))
            plt.savefig('debug.png')
            plt.close()

            xxx = 0


if __name__ == "__main__":
    create_splits()
    _test_splits()
    # _debug_dataset()
