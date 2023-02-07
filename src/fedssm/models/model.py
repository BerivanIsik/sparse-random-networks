import torch
import copy
from .utils import get_model


class FedModel:
    """
        Federated model.
    """

    def __init__(self, params, device=None):
        self.params = params
        self.device = device
        self.model, self.optimizer = get_model(params, device)
        self.model_size = self.compute_model_size()  # bit

        if self.params.get('model').get('optimizer').get('type') == 'ef_sign_sgd':
            self.beta_error = 0.0001
            self.e = dict()
            for k, val in self.model.named_parameters():
                self.e[k] = torch.zeros_like(val)

    def compute_model_size(self):
        """
            Assume torch.FloatTensor --> 32 bit
        """
        tot_params = 0
        for param in self.model.parameters():
            tot_params += param.numel()
        return tot_params * 32

    def inference(self, x_input):
        with torch.no_grad():
            self.model.eval()
            return self.model(x_input)

    def train_mask(self, data_loader):
        """
            Train the local masked model with the strategy adopted in
            https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
        """
        return self.perform_local_epochs(data_loader)

    def compute_delta(self, data_loader):
        """
            In case of SignSGD or Fedavg, compute the gradient of the local model.
        """

        delta = dict()
        xt = dict()
        for k, v in self.model.named_parameters():
            delta[k] = torch.zeros_like(v)
            xt[k] = copy.deepcopy(v)

        # Update local model
        loss = self.perform_local_epochs(data_loader)
        for k, v in self.model.named_parameters():
            delta[k] = v - xt[k]

        # Error compensation
        if self.params.get('model').get('optimizer').get('type') == 'ef_sign_sgd':
            with torch.no_grad():
                for k, v in self.model.named_parameters():
                    delta[k] += self.beta_error * self.e[k]
                    self.e[k] = delta[k] - torch.sign(delta[k])

        return loss, delta

    def perform_local_epochs(self, data_loader):
        """
            Compute local epochs, the training stategies depends on the adopted model.
        """
        loss = None
        for epoch in range(self.params.get('model').get('local_epochs')):
            running_loss = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            correct = 0
            for batch_idx, (train_x, train_y) in enumerate(data_loader):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                total += train_x.size(0)
                self.optimizer.zero_grad()
                y_pred = self.model(train_x, ths=None)
                loss = criterion(y_pred, train_y)
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                self.optimizer.step()
            if self.params.get('simulation').get('verbose'):
                train_loss = running_loss / total
                accuracy = correct / total
                print("Epoch {}: train loss {}  -  Accuracy {}".format(epoch+1, train_loss, accuracy))
        return loss

    def set_weights(self, w):
        self.model.load_state_dict(
            copy.deepcopy(w)
        )

    def get_weights(self):
        return self.model.state_dict()

    def save(self, folderpath):
        torch.save(self.model.state_dict(), folderpath.joinpath("local_model"))

    def load(self, folderpath):
        self.model.load_state_dict(torch.load(folderpath.joinpath("local_model"),
                                              map_location=torch.device('cpu')))


