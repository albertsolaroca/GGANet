import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

import sys
import json
import wandb

import torch.optim as optim

from utils.miscellaneous import read_config
from utils.miscellaneous import create_folder_structure
from utils.miscellaneous import initalize_random_generators
from utils.wandb_logger import log_wandb_data, save_response_graphs_in_ML_tracker
from utils.normalization import *
from utils.load import *

from training.train import training
from training.test import testing
from training.models import *

from training.visualization import plot_R2, plot_loss

# read config files
cfg = read_config("config_unrolling.yaml")
# create folder for result
exp_name = cfg['exp_name']
data_folder = cfg['data_folder']

res_columns = ['train_loss', 'valid_loss', 'test_loss', 'max_train_loss', 'max_valid_loss', 'max_test_loss',
               'min_train_loss', 'min_valid_loss', 'min_test_loss', 'r2_train', 'r2_valid',
               'r2_test', 'total_params', 'total_time', 'test_time']

all_wdn_names = cfg['networks']
initalize_random_generators(cfg, count=0)


def default_configuration():
    batch_size = cfg['trainParams']['batch_size']
    num_epochs = cfg['trainParams']['num_epochs']
    alpha = cfg['lossParams']['alpha']
    patience = cfg['earlyStopping']['patience']
    divisor = cfg['earlyStopping']['divisor']
    epoch_frequency = cfg['earlyStopping']['epoch_frequency']
    learning_rate = cfg['adamParams']['lr']
    weight_decay = cfg['adamParams']['weight_decay']
    algorithm = cfg['algorithms'][0]
    num_layers = cfg['hyperParams'][algorithm]['num_layers'][0]
    hid_channels = cfg['hyperParams'][algorithm]['hid_channels'][0]
    weight_decay = cfg['adamParams']['weight_decay']

    default_config = SimpleNamespace(batch_size=batch_size, num_epochs=num_epochs, alpha=alpha,
                                     patience=patience, divisor=divisor, epoch_frequency=epoch_frequency,
                                     algorithm=algorithm,
                                     num_layers=num_layers, hid_channels=hid_channels, learning_rate=learning_rate,
                                     weight_decay=weight_decay)

    return default_config


def make_config_dict(configuration):
    config = {}
    config['batch_size'] = configuration.batch_size
    config['num_epochs'] = configuration.num_epochs
    config['alpha'] = configuration.alpha
    config['patience'] = configuration.patience
    config['divisor'] = configuration.divisor
    config['epoch_frequency'] = configuration.epoch_frequency
    config['algorithm'] = configuration.algorithm
    config['num_layers'] = configuration.num_layers
    config['hid_channels'] = configuration.hid_channels
    config['learning_rate'] = configuration.learning_rate
    config['weight_decay'] = configuration.weight_decay

    return config


def prepare_training():
    print(f'\nWorking with {wdn}')
    if os.path.exists('prepared_data.pkl'):
        with open('prepared_data.pkl', 'rb') as file:
            prepared_data = pickle.load(file)
        tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn = prepared_data

    else:
        wdn = all_wdn_names[0]

        # retrieve wntr data
        tra_database, val_database, tst_database = load_raw_dataset(wdn, data_folder)
        # reduce training data
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
        gn = GraphNormalizer()
        gn = gn.fit(tra_dataset)

        tra_dataset, _ = create_dataset(tra_database, normalizer=gn)
        val_dataset, _ = create_dataset(val_database, normalizer=gn)
        tst_dataset, _ = create_dataset(tst_database, normalizer=gn)
        node_size, edge_size = tra_dataset[0].x.size(-1), tra_dataset[0].edge_attr.size(-1)
        # number of nodes
        junctions = (tra_database[0].node_type == 0).numpy().sum()
        tanks = (tra_database[0].node_type == 2).numpy().sum()
        output_nodes = junctions + tanks  # remove reservoirs
        # dataloader
        # transform dataset for MLP
        # We begin with the MLP versions, when I want to add GNNs, check Riccardo's code
        A10, A12 = create_incidence_matrices(tra_dataset, A12_bar)
        tra_dataset_MLP, num_inputs, indices = create_dataset_MLP_from_graphs(tra_dataset)
        val_dataset_MLP = create_dataset_MLP_from_graphs(val_dataset)[0]
        tst_dataset_MLP = create_dataset_MLP_from_graphs(tst_dataset)[0]

        prepared_data = (tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn)

        # Save the prepared data to a file
        with open('prepared_data.pkl', 'wb') as file:
            pickle.dump(prepared_data, file)

    return tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn


def train(configuration, tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn):
    # configuration = wandb.config
    tra_loader = torch.utils.data.DataLoader(tra_dataset_MLP,
                                             batch_size=configuration.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset_MLP,
                                             batch_size=configuration.batch_size, shuffle=False, pin_memory=True)
    tst_loader = torch.utils.data.DataLoader(tst_dataset_MLP,
                                             batch_size=configuration.batch_size, shuffle=False, pin_memory=True)
    # create results dataframe
    results_df = pd.DataFrame()

    # update wandb config
    algorithm = configuration.algorithm
    combination = {}
    combination['num_layers'] = configuration.num_layers
    combination['hid_channels'] = configuration.hid_channels
    combination['indices'] = indices
    combination['junctions'] = junctions
    combination['num_outputs'] = output_nodes

    # create folder structure
    results_folder = create_folder_structure("MLPs", algorithm, wdn,
                                             parent_folder='./experiments')

    # initialize pytorch device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    num_epochs = configuration.num_epochs
    alpha = configuration.alpha

    model, tra_losses, val_losses, elapsed_time = training(model, optimizer, tra_loader, val_loader,
                                                           patience=patience, report_freq=0,
                                                           n_epochs=num_epochs,
                                                           alpha=alpha, lr_rate=lr_rate, lr_epoch=lr_epoch,
                                                           normalization=None,
                                                           path=f'{results_folder}/{wdn}/{algorithm}/')
    loss_plot = plot_loss(tra_losses, val_losses, f'{results_folder}/{wdn}/{algorithm}/loss')
    R2_plot = plot_R2(model, val_loader, f'{results_folder}/{wdn}/{algorithm}/R2', normalization=gn)[1]

    # Logging plots on WandB
    # wandb.log({"Minimum validation loss": np.min(val_losses)})
    # wandb.log({"Loss": wandb.Image(loss_plot + ".png")})
    # wandb.log({"R2": wandb.Image(R2_plot + ".png")})
    # store training history and model
    pd.DataFrame(data=np.array([tra_losses, val_losses]).T).to_csv(
        f'{results_folder}/{wdn}/{algorithm}/hist.csv')
    torch.save(model, f'{results_folder}/{wdn}/{algorithm}/models.csv')

    # compute and store predictions, compute r2 scores
    losses = {}
    max_losses = {}
    min_losses = {}
    r2_scores = {}
    for split, loader in zip(['training', 'validation', 'testing'], [tra_loader, val_loader, tst_loader]):
        losses[split], max_losses[split], min_losses[split], pred, real, test_time = testing(model, loader,
                                                                                             normalization=gn)
        r2_scores[split] = r2_score(real, pred)
        pd.DataFrame(data=real.reshape(-1, output_nodes)).to_csv(
            f'{results_folder}/{wdn}/{algorithm}/pred/{split}/real.csv')  # save real obs

    # log_wandb_data(combination, wdn, algorithm, len(tra_database), len(val_database), len(tst_database),
    #                cfg, train_config, loss_plot, R2_plot)

    # store results
    results_df.loc[0, res_columns] = (losses['training'], losses['validation'], losses['testing'],
                                      max_losses['training'], max_losses['validation'],
                                      max_losses['testing'],
                                      min_losses['training'], min_losses['validation'],
                                      min_losses['testing'],
                                      r2_scores['training'], r2_scores['validation'], r2_scores['testing'],
                                      total_parameters, elapsed_time, test_time)
    # Saving configuration
    if isinstance(configuration, SimpleNamespace):
        with open(f'{results_folder}/{wdn}/{algorithm}/configuration.json', 'w') as fp:
            json.dump(vars(configuration), fp)
    else:
        config = make_config_dict(configuration)
        with open(f'{results_folder}/{wdn}/{algorithm}/configuration.json', 'w') as fp:
            json.dump(config, fp)

    _, _, _, pred, real, time = testing(model, val_loader)
    pred = gn.inverse_transform_array(pred, 'pressure')
    real = gn.inverse_transform_array(real, 'pressure')
    pred = pred.reshape(-1, output_nodes)
    real = real.reshape(-1, output_nodes)

    for i in [0, 1, 7, 36]:
        names = {0: 'Reservoir', 1: 'Next to Reservoir', 7: 'Random Node', 36: 'Tank'}
        save_response_graphs_in_ML_tracker(real, pred, names[i], i)

    # save graph normalizer
    # with open(f'{results_folder}/{wdn}/{algorithm}/gn.pickle', 'wb') as handle:
    #     pickle.dump(gn, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(f'{results_folder}/{wdn}/{algorithm}/model.pickle', 'wb') as handle:
    #     torch.save(model, handle)
    # results_df.to_csv(f'{results_folder}/{wdn}/{algorithm}/results_{algorithm}.csv')


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


# Main method
if __name__ == "__main__":
    agent = True
    if not agent:
        default_config = default_configuration()
        tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn = prepare_training()
        train(default_config, tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn)
    else:
        wandb.init()
        tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn = prepare_training()
        train(tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes, wdn)
        wandb.finish()
