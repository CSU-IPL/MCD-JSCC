from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
from torchvision.datasets import CIFAR10

from Get_datasets import *
import math
import torch.utils.data as data
NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        print(self.data_dir)
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img
    def __len__(self):
        return len(self.imgs)


def get_loader(config):

    data_root = getattr(config, 'data_root', os.environ.get('DATASET_ROOT', 'data'))

    if config.trainset == 'DIV2K':
        Img_Train = os.path.join(data_root, "DIV2K", "train")
        transform_train = transforms.Compose([
            transforms.RandomCrop((config.train_img_size, config.train_img_size)),
            transforms.ToTensor()
        ])
        train_dataset = OneImageDataset(
            dir=Img_Train,
            transform=transform_train
        )

    elif config.trainset == 'Kodak':
        Img_Train = os.path.join(data_root, "Kodak")
        transform_train = transforms.Compose([
            transforms.CenterCrop((config.train_img_size, config.train_img_size)),
            transforms.ToTensor()
        ])
        train_dataset = OneImageDataset(
            dir=Img_Train,
            transform=transform_train
        )
    elif config.trainset == 'CLIC':
        train_data_dir = [os.path.join(data_root, "CLIC2021")]
        train_dataset = Datasets(train_data_dir)

    elif config.trainset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dataset = CIFAR10(root=data_root,
                                train=True,
                                transform=transform_train,
                                download=False)

    elif config.trainset == 'CelebA':
        Img_Train = os.path.join(data_root, "CelebA", "train", str(config.train_img_size))
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = OneImageDataset(
            dir=Img_Train,
            transform=transform_train
        )

    train_loader = data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               drop_last=True)

    if config.testset == 'DIV2K':
        Img_Val = os.path.join(data_root, "DIV2K", "test")
        transform_test = transforms.Compose([
            transforms.CenterCrop((config.train_img_size, config.train_img_size)),
            transforms.ToTensor(),
        ])
        test_dataset = OneImageDataset(
            dir=Img_Val,
            transform=transform_test
        )

    elif config.testset == 'Chicken':
        Img_Val = os.path.join(data_root, "Chicken")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = OneImageDataset(
            dir=Img_Val,
            transform=transform_test
        )

    elif config.testset == 'Kodak':
        Img_Val = os.path.join(data_root, "Kodak")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = OneImageDataset(
            dir=Img_Val,
            transform=transform_test
        )
    elif config.testset == 'CLIC':
        test_data_dir = [os.path.join(data_root, "CLIC2020", "mobile_test", "mobile")]
        test_dataset = Datasets(test_data_dir)

    elif config.testset == 'CIFAR10':
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        test_dataset = CIFAR10(root=data_root,
                               train=False,
                               transform=transform_test,
                               download=False)

    elif config.testset == 'CelebA':
        Img_Val = os.path.join(data_root, "CelebA", "val", str(config.train_img_size))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = OneImageDataset(
            dir=Img_Val,
            transform=transform_test
        )

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=config.test_batch_size,
                                  shuffle=False)

    return train_loader, test_loader

