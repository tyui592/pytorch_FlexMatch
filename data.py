"""Data code."""

import torch
import random
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, STL10

from random import choices
from collections import defaultdict
from augmentation import Augmenter

MEANSTD = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}


def load_dataset(data='cifar10', split='train'):
    """Load a dataset."""
    root = f"./data/{data}"
    if 'cifar' in data:
        if split == 'train':
            split = True
        elif split == 'test':
            split = False

    if data == 'cifar10':
        dataset = CIFAR10(root=root, train=split, download=True)

    elif data == 'cifar100':
        dataset = CIFAR100(root=root, train=split, download=True)

    elif data == 'svhn':
        dataset = SVHN(root=root, split=split, download=True)

    elif data == 'stl10':
        dataset = STL10(root=root, split=split, download=True)

    return dataset


class Dataset:
    """Train Dataset."""

    def __init__(self, dataset, indices, num_iter=None):
        """Get dataset and item indices."""
        self.dataset = dataset
        self.indices = indices
        if num_iter is None:
            self.num_iter = len(self.indices)
        else:
            self.num_iter = num_iter

    def __len__(self):
        """length."""
        return self.num_iter

    def __getitem__(self, i):
        """getitem."""
        if self.num_iter != len(self.indices):
            i = random.randint(0, len(self.indices)-1)
        index = self.indices[i]
        img, label = self.dataset[index]
        return img, label, index


def split_train_datasets(data='cifar10',
                         num_X=250,
                         num_iter_X=64*1024,
                         num_iter_U=64*7*1024,
                         include_x_in_u=True):
    """Get X and U datasets.

    * Parameters
        - data: 'cifar10', 'cifar100', 'svhn' and 'stl10'.
        - num_X: number of labeled data.
        - include_x_in_u: include the labeled data in unlabeled data.
    """
    dataset = load_dataset(data, split='train')

    num_classes = 100 if data == 'cifar100' else 10

    if 'cifar' in data:
        labels = dataset.targets
    else:
        labels = dataset.labels

    idx_per_cls = defaultdict(list)
    for i, label in enumerate(labels):
        idx_per_cls[label].append(i)

    indices_x = []
    for k, v in idx_per_cls.items():
        indices_x += choices(v, k=(num_X//num_classes))

    X = Dataset(dataset, indices_x, num_iter_X)

    indices_u = list(range(len(dataset)))
    if not include_x_in_u:
        indices_u = list(set(indices_u) - set(indices_x))
    U = Dataset(dataset, indices_u, num_iter_U)

    return X, U


class PreProcessor:
    """Data Preprocessor."""

    def __init__(self, meanstd, policies):
        """Get augmentation policy and meanstd for normalization."""
        self.augs = [Augmenter(policy=policy) for policy in policies]
        mean, std = meanstd
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    def to_input(self, images):
        """Change from pil to tensor and normalize it."""
        x = torch.stack([self.to_tensor(image) for image in images])
        return x

    def __call__(self, batch):
        """Preprocess."""
        images, labels, indices = list(zip(*batch))

        label = torch.tensor(labels)
        index = torch.tensor(indices)

        tensors = []
        for aug in self.augs:
            x = self.to_input([aug(image) for image in images])
            tensors.append(x)
        return tensors, label, index


def get_dataloaders(data: str = 'cifar10', num_X: int = 250,
                    include_x_in_u=True, augs: list[int] = [1, 2],
                    batch_size: int = 64, mu: float = 7):
    """Get dataloaders."""
    # train dataset
    _X, _U = split_train_datasets(data=data,
                                  num_X=num_X,
                                  num_iter_X=batch_size*1024,
                                  num_iter_U=batch_size*mu*1024,
                                  include_x_in_u=include_x_in_u)

    train_processor = PreProcessor(policies=augs,
                                   meanstd=MEANSTD[data])

    X = DataLoader(dataset=_X,
                   batch_size=batch_size,
                   shuffle=True,
                   collate_fn=train_processor)

    U = DataLoader(dataset=_U,
                   batch_size=int(batch_size*mu),
                   shuffle=True,
                   collate_fn=train_processor)

    # test dataset
    test_processor = PreProcessor(policies=[0],
                                  meanstd=MEANSTD[data])
    test_data = load_dataset(data=data,
                      split='test')
    _T = Dataset(test_data, range(len(test_data)))
    T = DataLoader(dataset=_T,
                   batch_size=batch_size,
                   shuffle=False,
                   collate_fn=test_processor)

    return X, U, T
