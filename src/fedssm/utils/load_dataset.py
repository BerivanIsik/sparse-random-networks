from torch.utils.data import random_split
import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms

from pathlib import Path


def split_dataset(dataset: torch.utils.data.dataset, n_clients, lengths_list=None):
    """
        Return iid-split datasets
    """
    if lengths_list is None:
        int_lengths = int(len(dataset) / n_clients)
        lengths_list = [int_lengths]*(n_clients - 1)
        lengths_list.append(len(dataset) - (int_lengths * (n_clients - 1)))
    return random_split(dataset=dataset, lengths=lengths_list)


def get_mnist():
    """
        Return global train and test datasets for MNIST
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5),
         transforms.Lambda(lambda x: torch.flatten(x))])
    
    data_path = Path(__file__).parent.parent.parent.resolve().joinpath("data/mnist")
    global_train_dataset = datasets.MNIST(root=data_path,
                                          train=True,
                                          download=True,
                                          transform=transform)

    global_test_dataset = datasets.MNIST(root=data_path,
                                         train=False,
                                         download=True,
                                         transform=transform)
    
    return global_train_dataset, global_test_dataset


def get_cifar10():
    """
        Return global train and test datasets for CIFAR10
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])

    data_path = Path(__file__).parent.parent.parent.resolve().joinpath("data/cifar10")
    global_train_dataset = datasets.CIFAR10(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    global_test_dataset = datasets.CIFAR10(root=data_path,
                                           train=False,
                                           download=True,
                                           transform=transform)

    return global_train_dataset, global_test_dataset


dataset_dict = {
    "mnist": get_mnist,
    "cifar10": get_cifar10
}


def get_dataset(params):
    """
        Return the global train and test dataset
    """
    dataset_id = params.get('data').get('dataset')

    return dataset_dict[dataset_id]()

