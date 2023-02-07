import torch
from fedssm.simulator import Simulator
from fedssm.utils.read_data import read_params
from fedssm.utils.get_split_dataset import get_data_loaders

from pathlib import Path

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    current_folder = Path(__file__).parent.resolve()
    params_path = current_folder.joinpath('fedssm/params.yaml')

    params = read_params(params_path)
    client_train_loaders, server_test_loader = get_data_loaders(dataset=params.get('data').get('dataset'),
                                                                nclients=params.get('simulation').get('n_clients'),
                                                                batch_size=params.get('model').get('batch_size'),
                                                                classes_pc=params.get('data').get('classes_pc'),
                                                                split=params.get('data').get('split'))

    simulator = Simulator(params,
                          train_data=client_train_loaders,
                          test_data=server_test_loader,
                          device=device)

    for run in range(simulator.n_runs):
        simulator.run_folder = simulator.sim_folder.joinpath("run_{}".format(run))
        Path.mkdir(simulator.run_folder)
        simulator.start()
        simulator.fed_training()
