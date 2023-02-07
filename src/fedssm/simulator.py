import torch
from torch.utils.data import DataLoader
import numpy as np
from .server import Server
from .utils.load_dataset import split_dataset
from .utils.non_iid_cifar import get_data_loaders
from .utils.output import create_output_folder
from .client import Client
from torch.utils.tensorboard import SummaryWriter


class Simulator:
    """
        This is the class that handles the simulation flow.
    """

    def __init__(self, params, train_data, test_data, device=None):

        self.params = params
        self.device = device
        self.sim_folder = create_output_folder('fed', params=params)
        self.run_folder = None
        self.n_clients = self.params.get('simulation').get('n_clients')
        self.n_rounds = self.params.get('simulation').get('n_rounds')
        self.global_test_loader = test_data
        self.train_loaders = train_data
        self.n_runs = self.params.get("simulation").get("tot_sims")
        self.writer = None
        self.opt = self.params.get('model').get('optimizer').get('type')
        self.server = None
        self.clients_list = None
        self.n_sampled_clients = self.params.get('simulation').get('sampled_clients')

    def start(self):
        self.writer = SummaryWriter(log_dir=self.run_folder)
        self.server = Server(params=self.params, device=self.device)
        self.clients_list = [Client(self.params, dataset=local_data,
                                    initial_weights=self.server.global_model.get_weights(),
                                    device=self.device)
                             for local_data in self.train_loaders]
        self.server.set_client_list(self.clients_list)

    def local_training(self, sampled_clients, n_round):
        """
            Training the local model for each client
        """
        for client in sampled_clients:
            self.clients_list[client].train_local(n_round)
        # Aggregate the gradients at the server side
        p_round, bitrate_round, freq_round = self.server.aggregate_gradients(sampled_clients, n_round)
        print(f'Avg. p is {p_round}')
        print(f'Avg. bitrate is {bitrate_round}')
        print(f'Avg. frequency of 1s is {freq_round}')
        self.writer.add_scalar('Avg-Param', p_round, n_round)
        self.writer.add_scalar('Avg-Bitrate', bitrate_round, n_round)
        self.writer.add_scalar('Avg-Freq', freq_round, n_round)

    def fed_training(self):
        """
            Coordinate the federated training loop.
        """
        print('Start Simulation: ')
        print(f'N° of Clients: {self.n_clients}.')
        print(f'Sampling {self.n_sampled_clients} clients per round,')
        print(f'Model: {type(self.server.global_model.model)}.')
        print(f'Optimizer: {self.opt}.')
        print('Split: {}.'.format(self.params.get('data').get('split')))
        if self.params.get('data').get('split') == 'non-iid':
            print('Max number of classes per client: {}.'.format(
                self.params.get('data').get('classes_pc')))
        if self.params.get('model').get('optimizer').get('noisy'):
            print('Injects Gaussian noise with std {}.' .format(self.params.get('model').get('optimizer').get('std')))
        else:
            print('No noise injection.')

        for t in range(self.params.get('simulation').get('n_rounds') + 1):
            print("************ Round {} **************".format(t))

            # Sample a random subset of clients
            sampled_clients = self.server.sample_clients(n_samples=self.n_sampled_clients, t=t)

            # Broadcast the model to the sampled clients
            self.server.broadcast_model(sampled_clients=sampled_clients)

            # Train local models of the sampled clients
            self.local_training(sampled_clients=sampled_clients, n_round=t)

            # Testing
            loss, accuracy = self.test_global_model(n_samples=1)

            self.writer.add_scalar('Server-Loss/test', loss, t)
            self.writer.add_scalar('Server-Accuracy/test', accuracy, t)
            print("Test loss {} - Test accuracy {}".format(loss, accuracy))

            if not(t % 10):
                # Save the global model
                self.server.global_model.save(self.run_folder)
                self.writer.flush()
        self.writer.flush()
        self.writer.close()

    def test_global_model(self, n_samples=1, ths=None):
        """
            Test the global model on test dataset, n_samples to be used only with mask training.
        """
        total = 0
        correct = 0
        loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(self.global_test_loader):
                temp_l = []
                for n in range(n_samples):
                    test_x = test_x.to(self.device)
                    test_y = test_y.to(self.device)

                    outputs = self.server.global_model.model(test_x, ths=None)
                    _, pred_y = torch.max(outputs.data, 1)
                    temp_l.append(pred_y.cpu().numpy())
                axis = 0
                temp_l = np.asarray(temp_l)
                u, indices = np.unique(temp_l, return_inverse=True)
                maj_pred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(temp_l.shape),
                                                           None, np.max(indices) + 1), axis=axis)]

                total += test_y.size(0)
                maj_pred = np.asarray(maj_pred)
                correct += (maj_pred == test_y.cpu().numpy()).sum().item()
                loss += criterion(outputs, test_y)

        return loss / total, correct / total