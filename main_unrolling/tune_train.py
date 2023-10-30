import argparse
from types import SimpleNamespace

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

import sys
import pickle
import wandb

import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.utils import to_networkx
import torch.nn as nn
from torch.nn import Sequential, Linear
import networkx as nx
import wntr

from utils.miscellaneous import read_config
from utils.miscellaneous import create_folder_structure_MLPvsGNN
from utils.miscellaneous import initalize_random_generators
from utils.wandb_logger import log_wandb_data, save_response_graphs_in_ML_tracker
from utils.normalization import *
from utils.load import *

from training.train import training
from training.test import testing
from training.models import *


from training.visualization import plot_R2, plot_loss
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


# read config files
cfg = read_config("config_unrolling.yaml")
# create folder for result
exp_name = cfg['exp_name']
data_folder = cfg['data_folder']
results_folder = create_folder_structure_MLPvsGNN(cfg, parent_folder='./experiments')

all_wdn_names = cfg['networks']
initalize_random_generators(cfg, count=0)

# initialize pytorch device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# torch.set_num_threads(12)

# TO DO: at the moment I am not using the parsed values for batch size and num_epochs ;
# I am not using alpha as well because the loss has no "smoothness" penalty (yet)
batch_size = cfg['trainParams']['batch_size']
num_epochs = cfg['trainParams']['num_epochs']
alpha = cfg['lossParams']['alpha']
patience = cfg['earlyStopping']['patience']
divisor = cfg['earlyStopping']['divisor']
epoch_frequency = cfg['earlyStopping']['epoch_frequency']
learning_rate = cfg['adamParams']['lr']
weight_decay = cfg['adamParams']['weight_decay']

res_columns = ['train_loss', 'valid_loss', 'test_loss', 'max_train_loss', 'max_valid_loss', 'max_test_loss',
               'min_train_loss', 'min_valid_loss', 'min_test_loss', 'r2_train', 'r2_valid',
               'r2_test', 'total_params', 'total_time', 'test_time']

default_config = SimpleNamespace(batch_size=batch_size, num_epochs=num_epochs, alpha=alpha,
                                 patience=patience, divisor=divisor, epoch_frequency=epoch_frequency)



def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description="Hyper-parameters")
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='Batch size')
    # argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='Number of epochs')
    argparser.add_argument('--alpha', type=float, default=default_config.alpha, help='Alpha (smoothness penalty)')
    argparser.add_argument('--patience', type=int, default=default_config.patience, help='Patience')
    argparser.add_argument('--divisor', type=float, default=default_config.divisor, help='Divisor')
    argparser.add_argument('--epoch_frequency', type=int, default=default_config.epoch_frequency,
                           help='Epoch frequency')
    argparser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning Rate')
    argparser.add_argument('--weight_decay', type=float, default=weight_decay, help='Weight Decay')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def train(configuration):
    for ix_wdn, wdn in enumerate(all_wdn_names):
        print(f'\nWorking with {wdn}, network {ix_wdn + 1} of {len(all_wdn_names)}')

        # retrieve wntr data
        tra_database, val_database, tst_database = load_raw_dataset(wdn, data_folder)
        # reduce training data
        # tra_database = tra_database[:int(len(tra_database)*cfg['tra_prc'])]
        if cfg['tra_num'] < len(tra_database):
            tra_database = tra_database[:cfg['tra_num']]

        # remove PES anomaly
        if wdn == 'PES':
            if len(tra_database) > 4468:
                del tra_database[4468]
                print('Removed PES anomaly')
                print('Check', tra_database[4468].pressure.mean())

        # get GRAPH datasets
        # later on we should change this and use normal scalers from scikit
        tra_dataset, A12_bar = create_dataset(tra_database)
        checkpoint = tra_dataset[0]['x'][0]
        gn = GraphNormalizer()
        gn = gn.fit(tra_dataset)

        tra_dataset, _ = create_dataset(tra_database, normalizer=gn)
        val_dataset, _ = create_dataset(val_database, normalizer=gn)
        tst_dataset, _ = create_dataset(tst_database, normalizer=gn)
        node_size, edge_size = tra_dataset[0].x.size(-1), tra_dataset[0].edge_attr.size(-1)
        # number of nodes
        n_nodes = (tra_database[0].node_type == 0).numpy().sum() + (
                tra_database[0].node_type == 2).numpy().sum()  # remove reservoirs
        # dataloader
        # transform dataset for MLP
        # We begin with the MLP versions, when I want to add GNNs, check Riccardo's code
        A10, A12 = create_incidence_matrices(tra_dataset, A12_bar)
        tra_dataset_MLP, num_inputs, indices = create_dataset_MLP_from_graphs(tra_dataset)
        val_dataset_MLP = create_dataset_MLP_from_graphs(val_dataset)[0]
        tst_dataset_MLP = create_dataset_MLP_from_graphs(tst_dataset)[0]
        tra_loader = torch.utils.data.DataLoader(tra_dataset_MLP,
                                                 batch_size=configuration.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset_MLP,
                                                 batch_size=configuration.batch_size, shuffle=False, pin_memory=True)
        tst_loader = torch.utils.data.DataLoader(tst_dataset_MLP,
                                                 batch_size=configuration.batch_size, shuffle=False, pin_memory=True)
        # loop through different algorithms
        for algorithm in cfg['algorithms']:
            # Importing of configuration parameters
            hyperParams = cfg['hyperParams'][algorithm]
            all_combinations = ParameterGrid(hyperParams)

            # create results dataframe
            results_df = pd.DataFrame(list(all_combinations))
            results_df = pd.concat([results_df,
                                    pd.DataFrame(index=np.arange(len(all_combinations)),
                                                 columns=list(res_columns))], axis=1)

            for i, combination in enumerate(all_combinations):
                # wandb.init(reinit=True, name=wdn + "_" + algorithm + "_" + str(i))
                print(f'{algorithm}: training combination {i + 1} of {len(all_combinations)}\n')

                # update wandb config
                # wandb.config.update(combination)
                # wandb.config.update({'model': algorithm})

                combination['indices'] = indices
                print(indices)
                combination['num_outputs'] = n_nodes

                # model creation
                model = getattr(sys.modules[__name__], algorithm)(**combination).float().to(device)

                # get combination dictionary to determine how are indices made
                # print("Model", model, combination)

                total_parameters = sum(p.numel() for p in model.parameters())

                # model optimizer

                optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.999),
                                       lr=configuration.learning_rate, weight_decay=configuration.weight_decay)

                # training
                patience = configuration.patience
                lr_rate = configuration.divisor
                lr_epoch = configuration.epoch_frequency
                train_config = {"Patience": patience, "Learning Rate Divisor": lr_rate, "LR Epoch Division": lr_epoch}
                model, tra_losses, val_losses, elapsed_time = training(model, optimizer, tra_loader, val_loader,
                                                                       patience=patience, report_freq=0,
                                                                       n_epochs=num_epochs,
                                                                       alpha=alpha, lr_rate=lr_rate, lr_epoch=lr_epoch,
                                                                       normalization=None,
                                                                       path=f'{results_folder}/{wdn}/{algorithm}/')
                loss_plot = plot_loss(tra_losses, val_losses, f'{results_folder}/{wdn}/{algorithm}/loss/{i}')
                R2_plot = plot_R2(model, val_loader, f'{results_folder}/{wdn}/{algorithm}/R2/{i}', normalization=gn)[1]

                # Logging plots on WandB
                # wandb.log({"Loss": wandb.Image(loss_plot + ".png")})
                # wandb.log({"R2": wandb.Image(R2_plot + ".png")})
                # store training history and model
                pd.DataFrame(data=np.array([tra_losses, val_losses]).T).to_csv(
                    f'{results_folder}/{wdn}/{algorithm}/hist/{i}.csv')
                torch.save(model, f'{results_folder}/{wdn}/{algorithm}/models/{i}.csv')

                # compute and store predictions, compute r2 scores
                losses = {}
                max_losses = {}
                min_losses = {}
                r2_scores = {}
                for split, loader in zip(['training', 'validation', 'testing'], [tra_loader, val_loader, tst_loader]):
                    losses[split], max_losses[split], min_losses[split], pred, real, test_time = testing(model, loader,
                                                                                                         normalization=gn)
                    r2_scores[split] = r2_score(real, pred)
                    if i == 0:
                        pd.DataFrame(data=real.reshape(-1, n_nodes)).to_csv(
                            f'{results_folder}/{wdn}/{algorithm}/pred/{split}/real.csv')  # save real obs
                    pd.DataFrame(data=pred.reshape(-1, n_nodes)).to_csv(
                        f'{results_folder}/{wdn}/{algorithm}/pred/{split}/{i}.csv')

                # log_wandb_data(combination, wdn, algorithm, len(tra_database), len(val_database), len(tst_database),
                #                cfg, train_config, loss_plot, R2_plot)

                # store results
                results_df.loc[i, res_columns] = (losses['training'], losses['validation'], losses['testing'],
                                                  max_losses['training'], max_losses['validation'],
                                                  max_losses['testing'],
                                                  min_losses['training'], min_losses['validation'],
                                                  min_losses['testing'],
                                                  r2_scores['training'], r2_scores['validation'], r2_scores['testing'],
                                                  total_parameters, elapsed_time, test_time)

                _, _, _, pred, real, time = testing(model, val_loader)
                pred = gn.inverse_transform_array(pred, 'pressure')
                real = gn.inverse_transform_array(real, 'pressure')
                pred = pred.reshape(-1, n_nodes)
                real = real.reshape(-1, n_nodes)

                for i in [0, 1, 7, 36]:
                    names = {0: 'Reservoir', 1: 'Next to Reservoir', 7: 'Random Node', 36: 'Tank'}
                    save_response_graphs_in_ML_tracker(real, pred, names[i], i)

                # wandb.finish()
                # save graph normalizer
                # with open(f'{results_folder}/{wdn}/{algorithm}/gn.pickle', 'wb') as handle:
                #     pickle.dump(gn, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #
                # with open(f'{results_folder}/{wdn}/{algorithm}/model.pickle', 'wb') as handle:
                #     torch.save(model, handle)
                # results_df.to_csv(f'{results_folder}/{wdn}/{algorithm}/results_{algorithm}.csv')


# Main method
if __name__ == "__main__":
    parse_args()
    train(default_config)
