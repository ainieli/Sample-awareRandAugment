import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, SVHN


def load_cifar_data(ds_name, train_batch_size, test_batch_size, root='cifar', train_transform=None, test_transform=None,
                    use_proxy=False, num_proxy=4000, balanced=False):
    assert ds_name in ['CIFAR10', 'CIFAR100']

    if ds_name == 'CIFAR10':
        train_data = CIFAR10(root, train=True, download=True, transform=train_transform)
        test_data = CIFAR10(root, train=False, download=True, transform=test_transform)
    else:
        train_data = CIFAR100(root, train=True, download=True, transform=train_transform)
        test_data = CIFAR100(root, train=False, download=True, transform=test_transform)

    num_train = len(train_data)
    val_loader = None
    if use_proxy:
        if balanced:
            labels = [train_data[i][1] for i in range(num_train)]
            ss = StratifiedShuffleSplit(n_splits=1, test_size=num_proxy)
            _, idx = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]

            ss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
            sub_labels = [train_data[i][1] for i in idx]
            train_data_idx, val_data_idx = list(ss2.split(np.array(sub_labels)[:, np.newaxis], sub_labels))[0]
            train_data_idx = [idx[i] for i in train_data_idx]
            val_data_idx = [idx[i] for i in val_data_idx]
        else:
            idx = np.random.choice(range(num_train), size=num_proxy, replace=False)
            half_case = num_proxy // 2
            train_data_idx = np.random.choice(idx, size=half_case, replace=False)
            val_data_idx = [x for x in idx if x not in train_data_idx]

        val_data = [train_data[i] for i in val_data_idx]
        train_data = [train_data[i] for i in train_data_idx]

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=train_batch_size, shuffle=True,
            pin_memory=True, num_workers=2)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True,
        pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size,
        pin_memory=True, num_workers=2)

    return train_loader, val_loader, test_loader


def load_svhn_data(ds_name, train_batch_size, test_batch_size, root='svhn', train_transform=None, test_transform=None):
    assert ds_name in ['core', 'full']

    train_data = SVHN(root, split='train', download=True, transform=train_transform)
    test_data = SVHN(root, split='test', download=True, transform=test_transform)
    if ds_name == 'core':
        extra_data = SVHN(root, split='extra', download=True, transform=test_transform)
        train_data = train_data + extra_data
        print(len(train_data), len(test_data), len(extra_data))


    val_loader = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True,
        pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size,
        pin_memory=True, num_workers=2)

    return train_loader, val_loader, test_loader


# 120 classes used as proxy in DRA paper
def get_classes():
    return np.array(
        [620, 406, 973, 694, 871, 246, 261, 125, 36, 312, 58,
         644, 66, 101, 141, 744, 597, 165, 127, 472, 486, 659,
         727, 523, 119, 983, 310, 608, 823, 987, 1000, 366, 479,
         812, 345, 94, 660, 468, 524, 79, 402, 839, 542, 442,
         588, 138, 634, 504, 907, 178, 675, 943, 525, 151, 962,
         697, 565, 42, 349, 328, 493, 970, 267, 41, 158, 496,
         25, 906, 478, 671, 148, 801, 913, 776, 950, 354, 359,
         552, 448, 180, 520, 459, 770, 955, 289, 648, 609, 421,
         53, 265, 683, 902, 103, 306, 452, 957, 711, 314, 742,
         384, 695, 686, 28, 775, 625, 613, 255, 264, 899, 187,
         10, 614, 17, 984, 567, 161, 752, 971, 633, 510]) - 1


class ImageNetProxy(torch.utils.data.Dataset):

    def __init__(self, root, split, num_proxy_cls, num_proxy_per_cls, transform=None):
        assert split in ['train', 'val']
        if num_proxy_cls == 120:
            cls_idx = get_classes()
        else:
            # cls_idx = np.random.choice(range(1000), size=num_proxy_cls, replace=False)
            raise NotImplementedError('Currently only support 120 classes for the proxy dataset.')

        dir_root = os.path.join(root, split)
        dir_names = sorted(os.listdir(dir_root))
        self.img_paths = []
        self.labels = []
        for new_ci, ori_ci in enumerate(cls_idx):
            dir_name = dir_names[ori_ci]
            img_names = sorted(os.listdir(os.path.join(dir_root, dir_name)))
            for i in range(num_proxy_per_cls):
                self.img_paths.append(os.path.join(dir_root, dir_name, img_names[i]))
                self.labels.append(new_ci)
        self.transform = transform

    def __getitem__(self, index):
        img = plt.imread(self.img_paths[index])
        img = Image.fromarray(img).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)


def load_imagenet_data(train_batch_size, test_batch_size, root, train_transform=None, test_transform=None,
                       use_proxy=False, num_proxy_cls=120, num_proxy_per_cls=50):
    val_loader = None
    if use_proxy:
        train_data = ImageNetProxy(root, split='train', num_proxy_cls=num_proxy_cls,
                                   num_proxy_per_cls=num_proxy_per_cls, transform=train_transform)
        test_data = ImageNetProxy(root, split='val', num_proxy_cls=num_proxy_cls,
                                  num_proxy_per_cls=20, transform=train_transform)
        len_train = train_data.__len__()
        half_case = (num_proxy_cls * num_proxy_per_cls) // 2
        train_data_idx = np.random.choice(range(len_train), size=half_case, replace=False)
        val_data_idx = [x for x in range(len_train) if x not in train_data_idx]

        val_data = [train_data[i] for i in val_data_idx]
        train_data = [train_data[i] for i in train_data_idx]

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=train_batch_size, shuffle=True,
            pin_memory=True, num_workers=8)
    else:
        train_data = ImageNet(root, split='train', transform=train_transform)
        test_data = ImageNet(root, split='val', transform=test_transform)

        num_train = len(train_data)
        assert num_train == 1281167

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True,
        pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size,
        pin_memory=True, num_workers=8)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # load_cifar_data('CIFAR100', 1, 1, use_proxy=True)
    # load_cifar_data('CIFAR10', 1, 1, use_proxy=True, num_proxy=20, balanced=True)
    load_svhn_data('full', 1, 1)
