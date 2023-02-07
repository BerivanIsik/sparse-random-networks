"""
    Utility script to read parameters and data from files.
"""

import yaml


def read_params(filepath):
    """
        Read yaml file and return a Python dictionary with all parameters.
    """
    with open(filepath) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params
