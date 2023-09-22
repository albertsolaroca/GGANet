# Learning 

# Libraries
import time
import torch.nn as nn
import numpy as np

from main_unrolling.training.loss import *


def testing(model, loader, alpha=0, normalization=None):
    '''
    Function that tests a model and returns either the average losses or the predicted and real pressure values
    It can work both for ANN and GNN models (which require different DataLoaders)
    ------
    model: nn.Model
        e.g., GNN model
    loader: DataLoader
        data loader for dataset
    plot: bool
        if True, returns predicted and real pressure values
        if False, returns average losses
    alpha: float
        smoothness parameter (see loss.py for more info)
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})
    '''
    model.eval()
    losses = []
    pred = []
    real = []

    # retrieve model device (to correctly load data if GPU)
    device = next(model.parameters()).device

    # start measuring time
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            # if loop is needed to separate pytorch and pyg dataloaders
            if isinstance(loader, torch_geometric.loader.dataloader.DataLoader):
                # Load data to device
                real.append(batch.y)
                y = batch.y.to(device)

                # GNN model prediction
                out = model.double()(batch)
                pred.append(out.detach().cpu().numpy())

                # loss function = MSE if alpha=0
                loss = nn.MSELoss()(out, y.view(-1,1))

            elif isinstance(loader, torch.utils.data.dataloader.DataLoader):
                # Load data to device
                x, y = batch[0], batch[1]
                real.append(y)
                x = x.to(device).double()
                y = y.to(device).double()

                # ANN model prediction
                if len(y.shape) > 2:
                    out = model.double()(x, y.shape[1])
                else:
                    out = model.double()(x)
                pred.append(out.detach().cpu().numpy())

                # MSE loss function
                loss = nn.MSELoss()(out, y)

            # Normalization to have more representative loss values
            if normalization is not None:
                out = normalization.inverse_transform_array(pred[-1], 'pressure').flatten()
                y = normalization.inverse_transform_array(y.detach().cpu().numpy(), 'pressure').flatten()
                loss = nn.MSELoss()(out, y)

            losses.append(loss.cpu().detach())

        preds = np.concatenate(pred).reshape(-1, 1)
        reals = np.concatenate(real).reshape(-1, 1)
    elapsed_time = time.time() - start_time
    return np.array(losses).mean(), np.array(losses).max(),np.array(losses).min(), preds, reals, elapsed_time