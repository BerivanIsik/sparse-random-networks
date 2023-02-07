import math
import random
import numpy as np
import torch, torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from .load_dataset import split_dataset


def shuffle_list(data):

    """
        This function returns the shuffled data
    """

    for i in range(len(data)):
        tmp_len = len(data[i][0])
        index = [i for i in range(tmp_len)]
        random.shuffle(index)
        data[i][0], data[i][1] = shuffle_list_data(data[i][0],data[i][1])
    return data


def shuffle_list_data(x, y):
    """
        This function is a helper function, shuffles an
        array while maintaining the mapping between x and y
    """

    inds = list(range(len(x)))
    random.shuffle(inds)
    return x[inds], y[inds]


def get_cifar100(iid=False, transform=None):
    """
        Return CIFAR10 train/test data and labels as numpy arrays
    """

    data_train = torchvision.datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform)
    data_test = torchvision.datasets.CIFAR100('data/cifar100', train=False, download=True, transform=transform)

    if iid:
        return data_train, data_test

    x_train, y_train = data_train.data.transpose((0, 1, 2, 3)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 1, 2, 3)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_cifar10(iid=False, transform=None):
    """
        Return CIFAR10 train/test data and labels as numpy arrays
    """

    data_train = torchvision.datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform)
    data_test = torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True, transform=transform)

    if iid:
        return data_train, data_test

    x_train, y_train = data_train.data.transpose((0, 1, 2, 3)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 1, 2, 3)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_emnist(iid=False, transform=None):
    """
        Return global train and test datasets for EMNIST
    """

    data_path = Path(__file__).parent.parent.parent.resolve().joinpath("data/emnist")
    train_dataset = torchvision.datasets.EMNIST(root=data_path,
                                                train=True,
                                                download=True,
                                                transform=transform,
                                                split='balanced')

    test_dataset = torchvision.datasets.EMNIST(root=data_path,
                                               train=False,
                                               download=True,
                                               transform=transform,
                                               split='balanced')
    if iid:
        return train_dataset, test_dataset

    x_train, y_train = train_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(train_dataset.targets)
    x_test, y_test = test_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(test_dataset.targets)

    return x_train, y_train, x_test, y_test


def get_mnist(iid=False, transform=None):
    """
        Return global train and test datasets for MNIST
    """
    """transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5),
         transforms.Lambda(lambda x: torch.flatten(x))])"""

    data_path = Path(__file__).parent.parent.parent.resolve().joinpath("data/mnist")
    train_dataset = torchvision.datasets.MNIST(root=data_path,
                                               train=True,
                                               download=True,
                                               transform=transform)

    test_dataset = torchvision.datasets.MNIST(root=data_path,
                                              train=False,
                                              download=True,
                                              transform=transform)
    if iid:
        return train_dataset, test_dataset

    x_train, y_train = train_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(train_dataset.targets)
    x_test, y_test = test_dataset.data.cpu().numpy().transpose((0, 1, 2)), np.array(test_dataset.targets)

    return x_train, y_train, x_test, y_test


def clients_rand(train_len, n_clients):
    """
        This function creates a random distribution
        for the local datasets' size, i.e. number of images each client
        possess.
    """

    client_tmp = np.random.randint(10, 100, n_clients)
    sum_ = np.sum(client_tmp)
    clients_dist = (np.floor((client_tmp / sum_) * train_len)).astype(int)
    to_ret = list(clients_dist)
    to_ret[-1] += (train_len - clients_dist.sum())
    return to_ret


def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
    """
        Splits (data, labels) among 'n_clients s.t. every client can holds any number of classes which is trying to simulate real world dataset
        Input:
          data : [n_data x shape]
          labels : [n_data (x 1)] from 0 to n_labels(10)
          n_clients : number of clients
          verbose : True/False => True for printing some info, False otherwise
        Output:
          clients_split : splitted client data into desired format
    """

    def break_into(n, m):
        '''
        return m random integers with sum equal to n
        '''
        to_ret = [1 for i in range(m)]
        for i in range(n - m):
            ind = random.randint(0, m - 1)
            to_ret[ind] += 1
        return to_ret

    #### constants ####
    n_classes = len(set(labels))
    classes = list(range(n_classes))
    np.random.shuffle(classes)
    label_indcs = [list(np.where(labels == class_)[0]) for class_ in classes]

    #### classes for each client ####
    tmp = [np.random.randint(1, 10) for i in range(n_clients)]
    total_partition = sum(tmp)

    #### create partition among classes to fulfill criteria for clients ####
    class_partition = break_into(total_partition, len(classes))

    #### applying greedy approach first come and first serve ####
    class_partition = sorted(class_partition, reverse=True)
    class_partition_split = {}

    #### based on class partition, partitioning the label indexes ###
    for ind, class_ in enumerate(classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind], class_partition[ind])]

    #   print([len(class_partition_split[key]) for key in  class_partition_split.keys()])

    clients_split = []
    count = 0
    for i in range(n_clients):
        n = tmp[i]
        j = 0
        indcs = []

        while n > 0:
            class_ = classes[j]
            if len(class_partition_split[class_]) > 0:
                indcs.extend(class_partition_split[class_][-1])
                count += len(class_partition_split[class_][-1])
                class_partition_split[class_].pop()
                n -= 1
            j += 1

        ##### sorting classes based on the number of examples it has #####
        classes = sorted(classes, key=lambda x: len(class_partition_split[x]), reverse=True)
        if n > 0:
            raise ValueError(" Unable to fulfill the criteria ")
        clients_split.append([data[indcs], labels[indcs]])
    #   print(class_partition_split)
    #   print("total example ",count)

    clients_split = np.array(clients_split)

    return clients_split


def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True):
    """
        Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
        Input:
          data : [n_data x shape]
          labels : [n_data (x 1)] from 0 to n_labels
          n_clients : number of clients
          classes_per_client : number of classes per client
          shuffle : True/False => True for shuffling the dataset, False otherwise
        Output:
          clients_split : client data into desired format
    """

    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    clients_split = np.array(clients_split)

    return clients_split


class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        # self.inputs = torch.Tensor(inputs)
        self.inputs = inputs
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(dataset, train=True, verbose=False):

    if 'mnist' in dataset:
        transforms_train = {
            'general': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5),
                                           ])
        }
        transforms_eval = {
            'general': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5),
                                           ])
        }
    else:
        transforms_train = {
            'general': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
        }
        transforms_eval = {
            'general': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
        }
    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train['general'].transforms:
            print(' -', transformation)
        print()

    return transforms_train['general'], transforms_eval['general']


def get_data_loaders(dataset, nclients, batch_size, classes_pc=10, split='iid', real_wd=False):
    transforms_train, transforms_eval = get_default_data_transforms(dataset=dataset, verbose=True)

    if split == 'iid':
        data_train, data_test = dataset_dict[dataset](transform=transforms_train, iid=True)
        data_train_list = split_dataset(dataset=data_train, n_clients=nclients)
        data_loader_client_list = [DataLoader(local_data,
                                              batch_size=batch_size,
                                              shuffle=True) for local_data in data_train_list]
        data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)
        return data_loader_client_list, data_loader_test
    else:
        # Get data
        x_train, y_train, x_test, y_test = dataset_dict[dataset](iid=False)

        if real_wd:
            split = split_image_data_realwd(x_train, y_train, n_clients=nclients)
        else:
            split = split_image_data(x_train, y_train, n_clients=nclients,
                                     classes_per_client=classes_pc)

        split_tmp = shuffle_list(split)

        client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                      batch_size=batch_size, shuffle=True) for x, y in split_tmp]

        test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100,
                                                  shuffle=False)

    return client_loaders, test_loader


dataset_dict = {
    "mnist": get_mnist,
    "cifar10": get_cifar10,
    "cifar100": get_cifar100,
    "emnist": get_emnist
}
