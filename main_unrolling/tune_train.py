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

from utils.miscellaneous import read_config, create_folder_structure, initalize_random_generators
from utils.wandb_logger import save_response_graphs_in_ML_tracker, save_metric_graph_in_ML_tracker
from utils.normalization import *
from utils.load import *
from utils.metrics import calculate_metrics

from training.train import training
from training.test import testing
from training.models import *

from training.visualization import plot_R2, plot_loss


def default_configuration():
    # read config files
    cfg = read_config("config_unrolling.yaml")
    initalize_random_generators(cfg, count=0)
    batch_size = cfg['trainParams']['batch_size']
    network = cfg['network'][0]
    samples = cfg['tra_num']
    num_epochs = cfg['trainParams']['num_epochs']
    alpha = 0
    patience = cfg['earlyStopping']['patience']
    divisor = cfg['earlyStopping']['divisor']
    epoch_frequency = cfg['earlyStopping']['epoch_frequency']
    learning_rate = cfg['adamParams']['lr']
    weight_decay = cfg['adamParams']['weight_decay']
    algorithm = cfg['algorithms'][0]
    num_layers = cfg['hyperParams'][algorithm]['num_layers'][0]
    try:
        hid_channels = cfg['hyperParams'][algorithm]['hid_channels'][0]
    except KeyError:
        hid_channels = 0

    default_config = SimpleNamespace(network=network, samples=samples, batch_size=batch_size, num_epochs=num_epochs, alpha=alpha,
                                     patience=patience, divisor=divisor, epoch_frequency=epoch_frequency, algorithm=algorithm,
                                     num_layers=num_layers, hid_channels=hid_channels, learning_rate=learning_rate, weight_decay=weight_decay, clipper=100)

    return default_config


def make_config_dict(configuration):
    config = {}
    config['network'] = configuration.network
    config['samples'] = configuration.samples
    config['batch_size'] = configuration.batch_size
    config['num_epochs'] = configuration.num_epochs
    config['alpha'] = 0
    config['patience'] = configuration.patience
    config['divisor'] = configuration.divisor
    config['epoch_frequency'] = configuration.epoch_frequency
    config['algorithm'] = configuration.algorithm
    config['num_layers'] = configuration.num_layers
    config['hid_channels'] = configuration.hid_channels
    config['learning_rate'] = configuration.learning_rate
    config['weight_decay'] = configuration.weight_decay

    return config

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config



def prepare_training(network, samples):
    wdn = network

    print(f'\nWorking with {wdn}')
    if os.path.exists(wdn + '_prep_data.pkl'):
        with open(wdn + '_prep_data.pkl', 'rb') as file:
            prepared_data = pickle.load(file)
        tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, tanks, output_nodes, names = prepared_data

    else:
        # create folder for result
        data_folder = '../data_generation/datasets'
        # retrieve wntr data
        tra_database, val_database, tst_database = load_raw_dataset(wdn, data_folder)
        # reduce training data
        if samples < len(tra_database):
            tra_database = tra_database[:samples]

        # remove PES anomaly
        if wdn == 'PES':
            if len(tra_database) > 4468:
                del tra_database[4468]
                print('Removed PES anomaly')
                print('Check', tra_database[4468].pressure.mean())

        tra_dataset, A12_bar = create_dataset(tra_database)

        junctions = (tra_database[0].node_type == JUNCTION_TYPE).numpy().sum()
        tanks = (tra_database[0].node_type == TANK_TYPE).numpy().sum()
        output_nodes = len(tra_dataset[0].y[0])  # remove reservoirs

        # get GRAPH datasets
        # later on we should change this and use normal scalers from scikit
        gn = GraphNormalizer(junctions + tanks, output=['pressure', 'pump_flow'])
        gn = gn.fit(tra_dataset)

        tra_dataset, _ = create_dataset(tra_database, normalizer=gn)
        val_dataset, _ = create_dataset(val_database, normalizer=gn)
        tst_dataset, _ = create_dataset(tst_database, normalizer=gn)
        node_size, edge_size = tra_dataset[0].x.size(-1), tra_dataset[0].edge_attr.size(-1)
        # number of nodes

        # dataloader
        # transform dataset for MLP
        # We begin with the MLP versions, when I want to add GNNs, check Riccardo's code
        # A10, A12 = create_incidence_matrices(tra_dataset, A12_bar)
        tra_dataset_MLP, num_inputs, indices = create_dataset_MLP_from_graphs(tra_dataset)
        val_dataset_MLP = create_dataset_MLP_from_graphs(val_dataset)[0]
        tst_dataset_MLP = create_dataset_MLP_from_graphs(tst_dataset)[0]
        datasets_MLP = [tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP]

        node_names = tra_database[0].node_stores[0]["ID"]
        # edge_names = tra_database[0].edge_stores[0]["edge_ID"]
        node_types = tra_database[0].node_stores[0]["type"]
        edge_types = tra_database[0].edge_stores[0]["edge_type"]

        names = {}
        names["node_ids"] = node_names
        names["node_types"] = node_types
        # names["edge_ids"] = edge_names
        names["edge_types"] = edge_types

        prepared_data = (datasets_MLP, gn, indices, junctions, tanks, output_nodes, names)

        # Save the prepared data to a file
        # with open(wdn + '_prep_data.pkl', 'wb') as file:
        #     pickle.dump(prepared_data, file)

    return datasets_MLP, gn, indices, junctions, tanks, output_nodes, names

def prepare_scheduling(network):
    wdn = network

    data_folder = '../data_generation/datasets'
    # retrieve wntr data
    tst_database = load_raw_dataset_test(wdn, data_folder)
    # reduce data since we only need one sample
    tst_database = tst_database[:1]

    tst_dataset, A12_bar = create_dataset(tst_database)

    junctions = (tst_database[0].node_type == JUNCTION_TYPE).numpy().sum()
    tanks = (tst_database[0].node_type == TANK_TYPE).numpy().sum()
    output_nodes = len(tst_dataset[0].y[0])  # remove reservoirs

    node_heads = tst_dataset[0].node_stores[0]['x'].numpy()[:, 0]
    # get GRAPH datasets
    # later on we should change this and use normal scalers from scikit
    gn = GraphNormalizer(junctions + tanks, output=['pressure', 'pump_flow'])
    gn = gn.fit(tst_dataset)

    tst_dataset, _ = create_dataset(tst_database, normalizer=gn)
    # number of nodes
    node_size, edge_size = tst_dataset[0].x.size(-1), tst_dataset[0].edge_attr.size(-1)

    # dataloader
    # transform dataset for MLP
    # We begin with the MLP versions, when I want to add GNNs, check Riccardo's code
    # A10, A12 = create_incidence_matrices(tra_dataset, A12_bar)
    tst_dataset_MLP, num_inputs, indices = create_dataset_MLP_from_graphs(tst_dataset)

    node_names = tst_database[0].node_stores[0]["ID"]
    edge_names = tst_database[0].edge_stores[0]["edge_ID"]
    node_types = tst_database[0].node_stores[0]["type"]
    edge_types = tst_database[0].edge_stores[0]["edge_type"]

    names = {}
    names["node_ids"] = node_names
    names["node_heads"] = dict(zip(node_names, node_heads))
    names["node_types"] = node_types
    names["edge_ids"] = edge_names
    names["edge_types"] = edge_types

    return tst_dataset_MLP, gn, indices, junctions, tanks, output_nodes, names


def train(configuration, datasets_MLP, gn, indices, junctions, tanks, output_nodes, agent):
    # configuration = wandb.config
    tra_loader = torch.utils.data.DataLoader(datasets_MLP[0],
                                             batch_size=configuration.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(datasets_MLP[1],
                                             batch_size=configuration.batch_size, shuffle=False, pin_memory=True)
    tst_loader = torch.utils.data.DataLoader(datasets_MLP[2],
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
    wdn = configuration.network

    # create folder structure
    results_folder = create_folder_structure("MLPs", algorithm, wdn,
                                             parent_folder='./experiments')

    # initialize pytorch device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model creation
    model = getattr(sys.modules[__name__], algorithm)(**combination).float().to(device)
    if agent:
        wandb.watch(model, log='all')

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
    # if configuration.clipper:
    #     max_norm = configuration.clipper
    # else:
    #     max_norm = None
    alpha = 0

    model, tra_losses, val_losses, elapsed_time = training(model, optimizer, tra_loader, val_loader,
                                                           patience=patience, report_freq=0,
                                                           n_epochs=num_epochs, max_norm=None,
                                                           alpha=alpha, lr_rate=lr_rate, lr_epoch=lr_epoch,
                                                           normalization=None,
                                                           path=f'{results_folder}/{wdn}/{algorithm}/')
    loss_plot = plot_loss(tra_losses, val_losses, f'{results_folder}/{wdn}/{algorithm}/loss')
    R2_plot = plot_R2(model, val_loader, f'{results_folder}/{wdn}/{algorithm}/R2', normalization=gn)[1]

    if agent:
        wandb.unwatch(model)
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

    # store results
    res_columns = ['train_loss', 'valid_loss', 'test_loss', 'max_train_loss', 'max_valid_loss', 'max_test_loss',
                   'min_train_loss', 'min_valid_loss', 'min_test_loss', 'r2_train', 'r2_valid',
                   'r2_test', 'total_params', 'total_time', 'test_time']

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

    pred = gn.denormalize_multiple(pred, output_nodes)
    real = gn.denormalize_multiple(real, output_nodes)

    dummy = Dummy(junctions + tanks).evaluate(real)
    dict_metrics, dummy_scores, model_scores = calculate_metrics(real, dummy, pred)
    print(dict_metrics)
    # Logging plots on WandB
    if agent:
        for i in [0, 1, 7, 36]:
            names = {0: 'Reservoir', 1: 'Next to Reservoir', 7: 'Random Node', 36: 'Tank'}
            save_response_graphs_in_ML_tracker(real, pred, names[i], i)
        wandb.log({"min_val_loss": np.min(val_losses)})
        wandb.log({"Loss": wandb.Image(loss_plot + ".png")})
        wandb.log({"R2": wandb.Image(R2_plot + ".png")})
        wandb.log(dict_metrics)

        save_metric_graph_in_ML_tracker(dummy_scores, model_scores, "NSE")

    # save graph normalizer
    # with open(f'{results_folder}/{wdn}/{algorithm}/gn.pickle', 'wb') as handle:
    #     pickle.dump(gn, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    with open(f'{results_folder}/{wdn}/{algorithm}/model.pickle', 'wb') as handle:
        torch.save(model, handle)
    # results_df.to_csv(f'{results_folder}/{wdn}/{algorithm}/results_{algorithm}.csv')




# Main method
if __name__ == "__main__":
    agent = True
    if not agent:
        default_config = default_configuration()
        datasets_MLP, gn, indices, junctions, tanks, output_nodes, names = prepare_training(default_config.network, default_config.samples)
        train(default_config, datasets_MLP, gn, indices, junctions, tanks, output_nodes, agent)
    else:
        # initialize random generators for numpy and pytorch
        np.random.seed(4320)
        torch.manual_seed(3407)
        wandb.init()
        datasets_MLP, gn, indices, junctions, tanks, output_nodes, names = prepare_training(wandb.config.network, wandb.config.samples)
        train(wandb.config, datasets_MLP, gn, indices, junctions, tanks, output_nodes, agent)
        wandb.finish()
