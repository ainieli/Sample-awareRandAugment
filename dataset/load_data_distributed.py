import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, Food101


class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class LT_Dataset_Visualization(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path


def load_cifar_data(ds_name, train_batch_size, test_batch_size, root='cifar', train_transform=None, test_transform=None,
                    use_proxy=False, num_proxy=4000, use_multi_gpu=False, same_data_per_node=False):
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
        idx = np.random.choice(range(num_train), size=num_proxy, replace=False)
        half_case = num_proxy // 2
        train_data_idx = np.random.choice(idx, size=half_case, replace=False)
        val_data_idx = [x for x in idx if x not in train_data_idx]

        val_data = [train_data[i] for i in val_data_idx]
        train_data = [train_data[i] for i in train_data_idx]

        if use_multi_gpu:
            if same_data_per_node:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=1, rank=0)
            else:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = torch.utils.data.RandomSampler(val_data)

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=train_batch_size, sampler=val_sampler, shuffle=True,
            pin_memory=True, num_workers=2)

    if use_multi_gpu:
        if same_data_per_node:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=1, rank=0)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False,
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
                       use_proxy=False, num_proxy_cls=120, num_proxy_per_cls=50, use_multi_gpu=False, same_data_per_node=False):
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

        if use_multi_gpu:
            if same_data_per_node:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=1, rank=0)
            else:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = torch.utils.data.RandomSampler(val_data)

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=train_batch_size, sampler=val_sampler,
            pin_memory=True, num_workers=8)
    else:
        train_data = ImageNet(root, split='train', transform=train_transform)
        test_data = ImageNet(root, split='val', transform=test_transform)

        num_train = len(train_data)
        assert num_train == 1281167

    if use_multi_gpu:
        if same_data_per_node:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=1, rank=0)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False,
        pin_memory=True, num_workers=8)

    return train_loader, val_loader, test_loader


def load_food101_data(train_batch_size, test_batch_size, root, train_transform=None, test_transform=None,
                    use_multi_gpu=False, same_data_per_node=False):
    val_loader = None

    train_data = Food101(root, split='train', transform=train_transform, download=True)
    test_data = Food101(root, split='test', transform=test_transform, download=False)

    if use_multi_gpu:
        if same_data_per_node:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=1, rank=0)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False,
        pin_memory=True, num_workers=8)

    return train_loader, val_loader, test_loader


def load_imagenet_lt_data(train_batch_size, test_batch_size, root, train_transform=None, test_transform=None,
                    use_multi_gpu=False, same_data_per_node=False):
    val_loader = None

    train_txt_path = os.path.join(root, 'ImageNet_LT_train.txt')
    test_txt_path = os.path.join(root, 'ImageNet_LT_test.txt')
    train_data = LT_Dataset(root, txt=train_txt_path, transform=train_transform)
    test_data = LT_Dataset(root, txt=test_txt_path, transform=test_transform)

    if use_multi_gpu:
        if same_data_per_node:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=1, rank=0)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False,
        pin_memory=True, num_workers=8)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    load_cifar_data('CIFAR100', 1, 1, use_proxy=True)
