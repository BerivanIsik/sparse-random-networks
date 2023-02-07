import torch.optim

from ..models.masked_layers import *


def get_model(params, device):
    model, optimizer = None, None

    if params.get('model').get('id') == 'conv10':
        model = Mask10CNN(init=params.get('model').get('init'), device=device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.1)

    elif params.get('model').get('id') == 'conv6':
        model = Mask6CNN(init=params.get('model').get('init'), device=device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.1)

    elif params.get('model').get('id') == 'conv4':
        model = Mask4CNN(init=params.get('model').get('init'), device=device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.1)


    return model, optimizer
