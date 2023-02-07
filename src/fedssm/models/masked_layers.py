import torch.nn as nn
from ..utils.random import Bern
import torch.nn.functional as F
import torch
import numpy as np
import copy

activation_dict = {'relu': F.relu,
                   'sigmoid': F.sigmoid}


class MaskedLinear(nn.Linear):
    """
        Implementation of masked linear layer, with training strategy in
        https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
    """

    def __init__(self, in_features, out_features, init='ME_init', device=None, **kwargs):
        super(MaskedLinear, self).__init__(in_features, out_features, device=device, **kwargs)
        self.device = device
        arr_weights = None
        self.device = device
        self.init = init
        self.c = np.e * np.sqrt(1 / in_features)
        # Different weight initialization distributions
        if init == 'ME_init':
            arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
        elif init == 'ME_init_sym':
            arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
            arr_weights = np.triu(arr_weights, k=1) + np.tril(arr_weights)
        elif init == 'uniform':
            arr_weights = np.random.uniform(-self.c, self.c, size=(out_features, in_features)) * np.sqrt(3)
        elif init == 'k_normal':
            arr_weights = np.random.normal(0, self.c, size=(out_features, in_features))

        self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, device=self.device,
                                                dtype=torch.float))

        arr_bias = np.random.choice([-self.c, self.c], size=out_features)
        self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=False, device=self.device,
                                                dtype=torch.float))

        # Weights of Mask
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True, device=self.device))
        self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True, device=self.device))

    def forward(self, x, ths=None):
        if ths is None:
            # Generate probability of bernouli distributions
            s_m = torch.sigmoid(self.mask)
            s_b_m = torch.sigmoid(self.bias_mask)
            g_m = Bern.apply(s_m)
            g_b_m = Bern.apply(s_b_m)
        else:
            nd_w_mask = torch.sigmoid(self.mask)
            nd_b_mask = torch.sigmoid(self.bias_mask)
            g_m = torch.where(nd_w_mask > ths, 1, 0)
            g_b_m = torch.where(nd_b_mask > ths, 1, 0)

        # Compute element-wise product with mask
        effective_weight = self.weight * g_m
        effective_bias = self.bias * g_b_m

        # Apply the effective weight on the input data
        lin = F.linear(x, effective_weight.to(self.device), effective_bias.to(self.device))
        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.weight.sum(), torch.abs(self.weight).sum(),
                                                                    self.mask.sum() / prod)


class MaskedConv2d(nn.Conv2d):
    """
        Implementation of masked convolutional layer, with training strategy in
        https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, init='ME_init', device=None, **kwargs):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, device=device, **kwargs)
        self.device = device
        arr_weights = None
        self.init = init
        self.c = np.e * np.sqrt(1/(kernel_size**2 * in_channels))

        if init == 'ME_init':
            arr_weights = np.random.choice([-self.c, self.c],
                                           size=(out_channels, in_channels, kernel_size, kernel_size))
        elif init == 'uniform':
            arr_weights = np.random.uniform(-self.c, self.c,
                                            size=(out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(3)
        elif init == 'k_normal':
            arr_weights = np.random.normal(0, self.c ** 2, size=(out_channels, in_channels, kernel_size, kernel_size))

        self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, device=self.device,
                                                dtype=torch.float))

        arr_bias = np.random.choice([-self.c, self.c], size=out_channels)
        self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=False, device=self.device, dtype=torch.float))

        self.mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True, device=self.device))
        self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True, device=self.device))
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x, ths=None):

        if ths is None:
            # Generate probability of bernouli distributions
            s_m = torch.sigmoid(self.mask)
            s_b_m = torch.sigmoid(self.bias_mask)
            g_m = Bern.apply(s_m)
            g_b_m = Bern.apply(s_b_m)
        else:
            nd_w_mask = torch.sigmoid(self.mask)
            nd_b_mask = torch.sigmoid(self.bias_mask)
            g_m = torch.where(nd_w_mask > ths, 1, 0)
            g_b_m = torch.where(nd_b_mask > ths, 1, 0)

        effective_weight = self.weight * g_m
        effective_bias = self.bias * g_b_m
        # Apply the effective weight on the input data
        lin = self._conv_forward(x, effective_weight.to(self.device), effective_bias.to(self.device))

        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.weight.sum(), torch.abs(self.weight).sum(),
                                                                    self.mask.sum() / prod)


class Mask4CNN(nn.Module):
    """
            4Conv model studied in
            https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf for cifar10.
    """

    def __init__(self, init='me', activation='relu', device=None):
        super(Mask4CNN, self).__init__()
        self.activation = activation
        self.init = init
        self.conv1 = MaskedConv2d(1, 64, kernel_size=3, stride=1, padding='same', init=init, device=device)
        self.conv2 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding='same', init=init, device=device)
        self.conv3 = MaskedConv2d(64, 128, kernel_size=3, stride=1, padding='same', init=init, device=device)
        self.conv4 = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding='same', init=init, device=device)

        self.dense1 = MaskedLinear(6272, 256, init=init, device=device)
        self.dense2 = MaskedLinear(256, 256, init=init, device=device)
        self.dense3 = MaskedLinear(256, 10, init=init, device=device)

    def forward(self, x, ths=None):
        x = activation_dict[self.activation](self.conv1(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv2(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv3(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv4(x, ths)), kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = activation_dict[self.activation](self.dense1(x, ths))
        x = activation_dict[self.activation](self.dense2(x, ths))
        x = self.dense3(x, ths)
        return x

    def save(self, folderpath):
        torch.save(self.state_dict(), folderpath.joinpath(
            f"conv4_model_{self.init}"))


class Mask6CNN(nn.Module):
    """
            6Conv model studied in
            https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf for cifar10.
    """

    def __init__(self, init='me', activation='relu', device=None):
        super(Mask6CNN, self).__init__()
        self.activation = activation
        self.init = init
        self.conv1 = MaskedConv2d(3, 64, 3, init=init, device=device, stride=1, padding='same')
        self.conv2 = MaskedConv2d(64, 64, 3, init=init, device=device, stride=1, padding='same')
        self.conv3 = MaskedConv2d(64, 128, 3, init=init, device=device, stride=1, padding='same')
        self.conv4 = MaskedConv2d(128, 128, 3, init=init, device=device, stride=1, padding='same')
        self.conv5 = MaskedConv2d(128, 256, 3, init=init, device=device, stride=1, padding='same')
        self.conv6 = MaskedConv2d(256, 256, 3, init=init, device=device, stride=1, padding='same')

        self.dense1 = MaskedLinear(4096, 256, init=init, device=device)
        self.dense2 = MaskedLinear(256, 256, init=init, device=device)
        self.dense3 = MaskedLinear(256, 10, init=init, device=device)

    def forward(self, x, ths):
        x = activation_dict[self.activation](self.conv1(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv2(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv3(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv4(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv5(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv6(x, ths)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = activation_dict[self.activation](self.dense1(x, ths))
        x = activation_dict[self.activation](self.dense2(x, ths))
        x = self.dense3(x, ths)
        return x

    def save(self, folderpath):
        torch.save(self.state_dict(), folderpath.joinpath(
            f"conv6_model_{self.init}"))

    def get_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6,
                self.dense1, self.dense2, self.dense3]


class Mask10CNN(nn.Module):
    """
            8Conv model studied in https://arxiv.org/pdf/1911.13299.pdf, here for cifar100.
    """

    def __init__(self, init='me', activation='relu', device=None):
        super(Mask10CNN, self).__init__()
        self.activation = activation
        self.init = init
        self.conv1 = MaskedConv2d(3, 64, 3, init=init, device=device, stride=1, padding='same')
        self.conv2 = MaskedConv2d(64, 64, 3, init=init, device=device, stride=1, padding='same')
        self.conv3 = MaskedConv2d(64, 128, 3, init=init, device=device, stride=1, padding='same')
        self.conv4 = MaskedConv2d(128, 128, 3, init=init, device=device, stride=1, padding='same')
        self.conv5 = MaskedConv2d(128, 256, 3, init=init, device=device, stride=1, padding='same')
        self.conv6 = MaskedConv2d(256, 256, 3, init=init, device=device, stride=1, padding='same')
        self.conv7 = MaskedConv2d(256, 512, 3, init=init, device=device, stride=1, padding='same')
        self.conv8 = MaskedConv2d(512, 512, 3, init=init, device=device, stride=1, padding='same')
        self.conv9 = MaskedConv2d(512, 1024, 3, init=init, device=device, stride=1, padding='same')
        self.conv10 = MaskedConv2d(1024, 1024, 3, init=init, device=device, stride=1, padding='same')

        self.dense1 = MaskedLinear(1024, 256, init=init, device=device)
        self.dense2 = MaskedLinear(256, 256, init=init, device=device)
        self.dense3 = MaskedLinear(256, 100, init=init, device=device)

    def forward(self, x, ths):
        x = activation_dict[self.activation](self.conv1(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv2(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv3(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv4(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv5(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv6(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv7(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv8(x, ths)), kernel_size=2, stride=2)
        x = activation_dict[self.activation](self.conv9(x, ths))
        x = F.max_pool2d(activation_dict[self.activation](self.conv10(x, ths)), kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = activation_dict[self.activation](self.dense1(x, ths))
        x = activation_dict[self.activation](self.dense2(x, ths))
        x = self.dense3(x, ths)
        return x

    def save(self, folderpath):
        torch.save(self.state_dict(), folderpath.joinpath(
            f"conv8_model_{self.init}"))

    def get_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8,
                self.conv9, self.conv10, self.dense1, self.dense2, self.dense3]




