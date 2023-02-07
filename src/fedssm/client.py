import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import bernoulli
from .models.model import FedModel
from .models.pruned_layers import *
from .models.dense_layers import *
from .models.masked_layers import *
import copy
from collections import OrderedDict


class Client:
    """
        Class implementing local clients, with local datasets.
    """

    def __init__(self, params, dataset, initial_weights, device):
        self.params = params
        self.local_data_loader = dataset
        self.n = len(self.local_data_loader)
        # Local model at the client
        self.device = device
        self.local_model = FedModel(params=self.params, device=self.device)
        self.set_local_weights(initial_weights)
        self.accumulated_gradients = None
        self.delta = None
        self.model_size = self.local_model.model_size  # bit
        self.epsilon = 0.01

    def train_local(self, n_round):
        """
            Call the training phase of the local model.
        """
        loss = self.local_model.train_mask(data_loader=self.local_data_loader)

    def set_local_weights(self, w):
        self.local_model.set_weights(w)

    def get_local_weights(self):
        return self.local_model.get_weights()

    def upload_mask(self, n_samples):
        """
            Mask samples to be uploaded to the server.
        """
        param_dict = dict()
        num_params = 0.0
        num_ones = 0.0
        with torch.no_grad():
            for _ in range(n_samples):
                for k, v in self.local_model.get_weights().items():
                    if 'mask' in k:
                        theta = torch.sigmoid(v).cpu().numpy()
                        updates_s = bernoulli.rvs(theta)
                        updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
                        updates_s = np.where(updates_s == 1, 1-self.epsilon, updates_s)

                        # Keep track of the frequency of 1s.
                        num_params += updates_s.size
                        num_ones += np.sum(updates_s)

                        if param_dict.get(k) is None:
                            param_dict[k] = torch.tensor(updates_s, device=self.device)
                        else:
                            param_dict[k] += torch.tensor(updates_s, device=self.device)
                    else:
                        param_dict[k] = v
        local_freq = num_ones / num_params
        return param_dict, local_freq, num_params