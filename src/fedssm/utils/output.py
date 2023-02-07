"""
    Utility script to prepare the simulation output folder.
"""

import shutil
from pathlib import Path
import os
import pickle


def create_output_folder(system, params=None):
    """
        Create the simulation output folder, initialize it by copying `params_{}.yaml`
        and return the folder name
    """
    current_folder = Path(__file__).parent.resolve()
    model_dict = params.get('model')
    sim_name = system + '_' + params.get('data').get('dataset') + '_' + str(params.get('simulation').get('sampled_clients')) + '_' + \
               str(params.get('simulation').get('n_clients')) + '_' + model_dict.get('id') + '_' + model_dict.get('mode') +\
               '_' + model_dict.get('optimizer').get('type') + '_' + params.get('data').get('split')
    if params.get('data').get('split') == 'non-iid':
        sim_name += '_' + str(params.get('data').get('classes_pc'))
    folder_name = current_folder.parent.parent.resolve().joinpath('results', sim_name)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    shutil.copy2(current_folder.parent.joinpath(params.get('path').get('path')),
                 folder_name.joinpath(params.get('path').get('path')))
    return folder_name