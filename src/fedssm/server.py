import torch

import math
from .models.model import FedModel
from .drive import drive_compress, drive_decompress, drive_plus_compress, drive_plus_decompress
from .eden.eden import eden_builder
from .models.masked_layers import *
import numpy as np
from typing import List
from .client import Client
from scipy.stats import norm
import copy

class Server:
    """
        Class for the central authority, i.e., the server, that coordinates the federated process.
    """

    def __init__(self, params, device):

        self.params = params
        self.n_clients = self.params.get('simulation').get('n_clients')
        self.clients_list = None
        self.device = device
        self.global_model = FedModel(self.params, device=self.device)
        self.stale_steps = 0
        self.update_number = 0
        self.stale_ths = 10

        if self.params.get('model').get('mode') == "mask":
            self.alphas = dict()
            self.betas = dict()
            self.lambda_init = 1
            for k, val in self.global_model.model.named_parameters():
                self.alphas[k] = torch.ones_like(val) * self.lambda_init
                self.betas[k] = torch.ones_like(val) * self.lambda_init

    def sample_clients(self, n_samples, t=None):
        """
            Sample clients at each round (now uniformly at random, can be changed).
        """
        return np.random.choice(np.arange(self.n_clients),
                                size=n_samples,
                                replace=False)

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params

    def set_client_list(self, clients_list: List[Client]):
        self.clients_list = clients_list

    def compute_n_mask_samples(self, n_round):
        """
            Return how many per-client samples, i.e., bits, the server wants to receive
            (can be a function of the round number). Now set to 1.
        """
        return 1

    def aggregate_gradients(self, client_idxs, n_roud):
        return self.sample_mask_aggregation(client_idxs, n_roud)

    def sample_mask_aggregation(self, client_idxs, n_roud):
        """
            Aggregation method for the federated sampling subnetworks scheme ("Bayesian Aggregation").
        """

        aggregated_weights = copy.deepcopy(self.global_model.get_weights())
        aggregated_p = dict()

        for k, v in self.global_model.model.named_parameters():
            if 'mask' in k:
                aggregated_p[k] = torch.zeros_like(v)

        with torch.no_grad():
            # Reset aggregation priors.
            # self.update_prior(n_roud)
            self.reset_prior()
            n_samples = self.compute_n_mask_samples(n_roud)
            p_update = []
            avg_bitrate = 0
            avg_freq = 0
            for client in client_idxs:
                sampled_mask, local_freq, num_params = self.clients_list[client].upload_mask(
                    n_samples=n_samples)
                avg_freq += local_freq
                local_bitrate = self.find_bitrate([local_freq + 1e-50, 1 - local_freq + 1e-50], num_params) + math.log2(num_params)
                avg_bitrate += local_bitrate / num_params
                for k, v in sampled_mask.items():
                    if 'mask' in k:
                        self.alphas[k] += v
                        self.betas[k] += (n_samples-v)
                        # Add layerwise estimated ps for each client
                        p_update.extend(v.cpu().numpy().flatten()/n_samples)
            avg_bitrate = avg_bitrate / len(client_idxs)
            avg_freq = avg_freq / len(client_idxs)
            # Update the posterior, and compute the mode of the beta distribution, as suggested in
            # https://neurips2021workshopfl.github.io/NFFL-2021/papers/2021/Ferreira2021.pdf
            for k, val in aggregated_weights.items():
                if 'mask' in k:
                    avg_p = (self.alphas[k] - 1) / (self.alphas[k] + self.betas[k] - 2)
                    if self.params.get('model').get('optimizer').get('noisy'):
                        avg_p = self.correct_bias(avg_p)
                    aggregated_weights[k] = torch.tensor(
                        torch.log(avg_p / (1 - avg_p)),
                        requires_grad=True,
                        device=self.device)
        self.global_model.set_weights(aggregated_weights)
        return np.mean(p_update), avg_bitrate, avg_freq

    def update_prior(self, n_round):
        """
            Compute when resetting the prior depending on the round number.
        """
        if n_round < 15:
            self.reset_prior()
        elif n_round % 5 == 0:
            self.reset_prior()

    def reset_prior(self):
        """
            Reset to uniform prior, depending on lambda_init.
        """
        self.alphas = dict()
        self.betas = dict()
        for k, val in self.global_model.model.named_parameters():
            self.alphas[k] = torch.ones_like(val) * self.lambda_init
            self.betas[k] = torch.ones_like(val) * self.lambda_init

    def broadcast_model(self, sampled_clients):
        """
            Send the global updated model to the clients.
        """
        for client in sampled_clients:
            self.clients_list[client].set_local_weights(self.global_model.get_weights())
