# Learning 

# Libraries
import time
import torch.nn as nn
import numpy as np
import torch
import torch_geometric
import torch.optim as optim
from tqdm import tqdm

from main_unrolling.training.loss import *
from main_unrolling.training.test import testing
from main_unrolling.utils.visualization import *

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_epoch(model, loader, optimizer, alpha=0, normalization=None, device=None):
    '''
    Function that trains a model for one iteration
    It can work both for ANN and GNN models (which require different DataLoaders)
    ------
    model: nn.Model
        e.g., GNN model
    loader: DataLoader
        data loader for dataset
    optimizer: torch.optim
        model optimizer (e.g., Adam, SGD)
    alpha: float
        smoothness parameter (see loss.py for more info)
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})
    '''
    model.train()
    losses = []

    # retrieve model device (to correctly load data if GPU)
    if device is None:
        device = next(model.parameters()).device

    for batch in loader:

        if isinstance(loader, torch_geometric.loader.dataloader.DataLoader):
            # Load data to device
            batch = batch.to(device)

            # Model prediction
            preds = model.to(device)(batch)

            # loss function = MSE if alpha=0
            # loss = smooth_loss(preds, batch, alpha=alpha)
            loss = nn.MSELoss()(preds, batch.y.double().to(device).view(-1,1))

        elif isinstance(loader, torch.utils.data.dataloader.DataLoader):
            # Load data to device
            x, y = batch[0], batch[1]
            x = x.to(device).double()
            y = y.to(device).double()

            # Model prediction
            preds = model.double()(x)

            # MSE loss function
            loss = nn.MSELoss()(preds, y)

        losses.append(loss.cpu().detach())

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return np.array(losses).mean()


def training(model, optimizer, train_loader, val_loader,
             n_epochs, patience=10, report_freq=100, alpha=0, lr_rate=10, lr_epoch=5000, normalization=None,
             device=None, path = None):
    '''
    Training function which returns the training and validation losses over the epochs
    Learning rate scheduler and early stopping routines working correctly
    ------
    model: nn.Model
        e.g., GNN model
    optimizer: torch.optim
        model optimizer (e.g., Adam, SGD)
    *_loader: DataLoader
        data loader for training, validation and testing
    n_epochs: int
        maximum number of total epochs
    patience: int
        number of subsequent occurrences where the validation loss is increasing
    report_freq: int
        printing interval
    alpha: float
        smoothness parameter (see loss.py for more info)
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})
    '''
    # create vectors for the training and validation loss
    train_losses = []
    val_losses = []

    # start measuring time
    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience, delta=1e-3, path=path + 'checkpoint.pt')

    # torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(1, n_epochs + 1)):
        # Model training
        train_loss = train_epoch(model, train_loader, optimizer, alpha=alpha, normalization=normalization,
                                 device=device)

        # Model validation
        val_loss, _, _, _, _, _ = testing(model, val_loader, alpha=alpha, normalization=normalization)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # learning rate scheduler
        if epoch % lr_epoch == 0:
            learning_rate = optimizer.param_groups[0]['lr'] / lr_rate
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print("Learning rate is divided by ", lr_rate, "to:", learning_rate)

        #Routine for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        # Print loss
        if epoch % report_freq == 0:
            print("epoch:", epoch,
                  "\t train MSE:", np.round(train_loss, 4),
                  "\t val MSE:", np.round(val_loss, 4)
                  )

    elapsed_time = time.time() - start_time

    return model, train_losses, val_losses, elapsed_time