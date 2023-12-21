import numpy as np
import torch
import yaml
import os
from itertools import product

def read_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''   
    
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    assert set(cfg['algorithms']).issubset(set(cfg['hyperParams'].keys())),\
        "There is an algorithm without hyperparameters"
        
    return cfg

def create_folder_structure(exp_name, algorithm, network, parent_folder='./results', max_trials=1000):
    '''
    ad hoc solution that has to be removed (e.g., add a split arg to original function for shortcut? 
                                                  long term you must redo all this)
    '''
    folder_name = exp_name
    # retrieve here list of architectures
    algorithms = [algorithm]
    networks = [network]
    
    create_folder_flag = True
    counter = 0
    while create_folder_flag == True:
        if counter == 0:
            suffix = ''    
        else:
            suffix = f'{counter:04d}'        
        results_folder =  f'{parent_folder}/{folder_name}{suffix}'
        if not os.path.exists(results_folder):                
            # creating folders
            print(f'Creating folder: {results_folder}')
            for wdn in networks:
                os.makedirs(f'{results_folder}/{wdn}', exist_ok=True)
                for algorithm in algorithms:                    
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}')
                    # os.makedirs(f'{results_folder}/{wdn}/{algorithm}/hist')
                    # os.makedirs(f'{results_folder}/{wdn}/{algorithm}/models')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/pred/')
                    # os.makedirs(f'{results_folder}/{wdn}/{algorithm}/loss/')
                    # os.makedirs(f'{results_folder}/{wdn}/{algorithm}/R2/')
                    for split in ['training','validation','testing']:
                        os.makedirs(f'{results_folder}/{wdn}/{algorithm}/pred/{split}')                    
            create_folder_flag = False
        else:
            counter += 1
            if counter > max_trials:
                raise OSError(f"Too many folders for experiment {folder_name}. Try changing it!")
    
    return results_folder


def create_folder_structure_MLPvsGNN(cfg, parent_folder='./results', max_trials=1000):
    '''
    ad hoc solution that has to be removed (e.g., add a split arg to original function for shortcut?
                                                  long term you must redo all this)
    '''
    folder_name = cfg['exp_name']
    # retrieve here list of architectures
    algorithms = cfg['algorithms']
    networks = cfg['network']

    create_folder_flag = True
    counter = 0
    while create_folder_flag == True:
        if counter == 0:
            suffix = ''
        else:
            suffix = f'{counter:04d}'
        results_folder = f'{parent_folder}/{folder_name}{suffix}'
        if not os.path.exists(results_folder):
            # creating folders
            print(f'Creating folder: {results_folder}')
            for wdn in networks:
                os.makedirs(f'{results_folder}/{wdn}', exist_ok=True)
                for algorithm in algorithms:
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/hist')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/models')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/pred/')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/loss/')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/R2/')
                    for split in ['training', 'validation', 'testing']:
                        os.makedirs(f'{results_folder}/{wdn}/{algorithm}/pred/{split}')
            create_folder_flag = False
        else:
            counter += 1
            if counter > max_trials:
                raise OSError(f"Too many folders for experiment {folder_name}. Try changing it!")

    return results_folder


def initalize_random_generators(cfg, count=0):
    '''
    This function initialites the random seeds specified in the config file.
    ------
    cfg: dict
        configuration file obtained with read_config
    count: int
        select seed used for testing
    '''
    # initialize random seeds for reproducibility
    np_seed=cfg['seeds']['np']
    torch_seed=cfg['seeds']['torch']
    
    # initialize random generators for numpy and pytorch
    np.random.seed(np_seed)       
    torch.manual_seed(torch_seed)
    
    return None    

def read_hyperparameters(cfg: dict, model: str):
    '''
    This function selects the hyperparameters specified in the config file.
    returns a list with all hyperparameters combinations
    ------
    cfg: dict
        configuration file obtained with read_config
    model: str
        'GNN' or 'ANN'. 
    '''
    if model == 'GNN':
        hid_channels = cfg['GNN_hyperp']['hid_channels']
        edge_channels = cfg['GNN_hyperp']['edge_channels']
        K = cfg['GNN_hyperp']['K']
        dropout_rate = cfg['GNN_hyperp']['dropout_rate']
        weight_decay = cfg['GNN_hyperp']['weight_decay']
        learning_rate = cfg['GNN_hyperp']['learning_rate']
        batch_size = cfg['GNN_hyperp']['batch_size']
        alpha = cfg['GNN_hyperp']['alpha']
        num_epochs = cfg['GNN_hyperp']['num_epochs']
        
        combinations = list(product(*[hid_channels, edge_channels, K, dropout_rate, weight_decay, learning_rate, batch_size, alpha, num_epochs]))
    
    elif model == 'ANN':
        hid_channels = cfg['ANN_hyperp']['hid_channels']
        hid_layers = cfg['ANN_hyperp']['hid_layers']
        dropout_rate = cfg['ANN_hyperp']['dropout_rate']
        weight_decay = cfg['ANN_hyperp']['weight_decay']
        learning_rate = cfg['ANN_hyperp']['learning_rate']
        batch_size = cfg['ANN_hyperp']['batch_size']
        num_epochs = cfg['ANN_hyperp']['num_epochs']
        
        combinations = list(product(*[hid_channels, hid_layers, dropout_rate, weight_decay, learning_rate, batch_size, num_epochs]))
        
    else:
        raise("model must be either 'GNN' or 'ANN'")

    return combinations


# Convert the Data object to a NetworkX graph and visualize using Matplotlib
def print_graph():
    graph = torch_geometric.utils.to_networkx(tra_dataset[0])
    pos = nx.spring_layout(graph, seed=42)  # Position the nodes for visualization
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_color='black')
    plt.title("Graph Visualization")
    plt.show()