import torch.nn as nn
from ..utils.random import Bern
import torch.nn.functional as F
import torch


class Dense4CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense4CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding='same', bias=False, device=device)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False, device=device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=False, device=device)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False, device=device)

        self.dense1 = nn.Linear(6272, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 10, device=device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Dense6CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense6CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same', device=device)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', device=device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', device=device)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', device=device)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', device=device)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', device=device)

        self.dense1 = nn.Linear(4096, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 10, device=device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Dense10CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same', device=device)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', device=device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', device=device)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', device=device)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', device=device)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', device=device)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same', device=device)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', device=device)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding='same', device=device)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding='same', device=device)

        self.dense1 = nn.Linear(1024, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 100, device=device)

    def forward(self, x, ths=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv9(x))
        x = F.max_pool2d(F.relu(self.conv10(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
