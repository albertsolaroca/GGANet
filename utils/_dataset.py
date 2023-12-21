# Libraries
import torch
import numpy as np
from itertools import product  
from torch_geometric.data import Data
from sklearn.utils import shuffle
from utils.scaling import *

# Dataset split: train, validate, test
def train_val_test(Dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    '''
    Split the dataset into training, validation and testing
    ------
    Dataset:list
        list of samples
    *_split: float
        percentage of data in each dataset
        the sum must be equal to 1
    '''
    assert train_split + val_split + test_split == 1, "The sum of train_split, val_split, and test_split must be 1"

    N_datasets = len(Dataset)
    np.random.seed(42)

    Dataset = shuffle(Dataset)

    train_dataset = Dataset[:round(N_datasets*train_split)]
    val_dataset = Dataset[round(N_datasets*train_split):(round(N_datasets*train_split)+round(N_datasets*val_split))]
    test_dataset = Dataset[(round(N_datasets*train_split)+round(N_datasets*val_split)):]

    return train_dataset, val_dataset, test_dataset


# Create geometric dataset
def create_dataset(database, normalization=None):
    '''
	Creates the true dataset from the pickle databases	
    ------    
	database : list
		each element in the list is a pickle file containing Data objects
    normalization: dict
        normalize the dataset using mean and std
	'''
    # Roughness info (Hazen-Williams)
    minR = 60 # (steel)
    maxR = 150 # (plastic)

    graphs = []
    
    if normalization is not None:
        elevation_mean, elevation_std = normalization['elevation']
        base_demand_mean, base_demand_std = normalization['demand']
        length_mean, length_std = normalization['length']
        pressure_max = normalization['pressure']
        
        for i in database:
            graph = Data()

            # Node attributes
            # elevation_head = i.elevation + i.base_head
            elevation_head = i.elevation
            elevation_head[elevation_head==0] = elevation_head.mean()
            
            elevation = (elevation_head-min(elevation_head)-elevation_mean)/elevation_std
            # elevation = (elevation_head-min(elevation_head))/(max(elevation_head)-min(elevation_head))
            base_demand = (i.base_demand-base_demand_mean)/base_demand_std
            graph.x = torch.stack((elevation, base_demand, i.type_1H), dim=1)

            # Position and ID
            graph.pos = i.pos
            graph.ID = i.ID
            
            # Edge index (Adjacency matrix)
            graph.edge_index = i.edge_index

            # Edge attributes
            diameter = i.diameter
            length = (i.length-length_mean)/length_std
            roughness = ((i.roughness) - minR)/(maxR - minR)
            graph.edge_attr = torch.stack((diameter, length, roughness), dim=1).float()

            # Graph output (pressure)
            pressure = i.pressure/pressure_max
            graph.y = pressure.reshape(-1,1)

            graphs.append(graph)
                    
    else:
        for i in database:
            graph = Data()

            # Node attributes
            # elevation_head = i.elevation + i.base_head
            elevation_head = i.elevation
            elevation_head[elevation_head==0] = elevation_head.mean()
            
            elevation = elevation_head-min(elevation_head)
            base_demand = i.base_demand*1000 #convert to l/s
            graph.x = torch.stack((elevation, base_demand, i.type_1H), dim=1)

            # Position and ID
            graph.pos = i.pos
            graph.ID = i.ID   
            
            # Edge index (Adjacency matrix)
            graph.edge_index = i.edge_index

            # Edge attributes
            diameter = i.diameter
            length = i.length
            roughness = i.roughness
            graph.edge_attr = torch.stack((diameter, length, roughness), dim=1).float()

            # Graph output (pressure)
            pressure = i.pressure
            graph.y = pressure.reshape(-1,1)

            graphs.append(graph)
            
    return graphs

        
        
# Create dataset for MLP
def create_dataset_ANN(database, normalization=None, features=['diameter','base_demand','diameter']):
    '''
    Creates the true dataset from the pickle databases
    ------
    database : list
        each element in the list is a pickle file containing Data objects
    normalization: dict
        normalize the dataset using mean and std
    '''
    
    # index edges to avoid duplicates: this considers all graphs to be UNDIRECTED!
    ix_edge = database[0].edge_index.numpy().T
    ix_edge = (ix_edge[:,0]<ix_edge[:,1])
    
    for ix_feat,feature in enumerate(features):
        for ix_item, item in enumerate(database):
            x_ = getattr(item,feature)
            if feature in ['diameter','roughness']:
                # check needed to avoid duplicates
                x_ = x_[ix_edge]
            elif feature not in ['base_demand']:
                raise ValueError(f'Feature {feature} not supported.')
            if ix_item == 0:
                x = x_
            else:
                x = torch.cat((x,x_), dim=0)            
        if ix_feat == 0:
            X = x.reshape(len(database),-1)
        else:
            X = torch.cat((X,x.reshape(len(database),-1)), dim=1)
    
    for ix_item, item in enumerate(database):
        if ix_item == 0:
            y = item.pressure
        else:
            y = torch.cat((y,item.pressure), dim=0)
    y = y.reshape(len(database),-1)
    
    if normalization is not None:
        pressure_max = normalization['pressure']
        y = y/pressure_max        
        
    return X, y
    
    '''
    diameters = database[0].diameters
    pressure = database[0].pressure

    for i in database[1:]:
        diameters = torch.cat((diameters,i.diameters), dim=0)
        pressure = torch.cat((pressure, i.pressure), dim=0)

    diameters = diameters.reshape(len(database), -1)    
    pressure = pressure.reshape(len(database), -1)  

    if normalization is not None:
        pressure_max = normalization['pressure']
        pressure = pressure/pressure_max        
        
    print('\nDiameters:\t',diameters.shape)
    print('\nPressure:\t',pressure.shape)

    return diameters, pressure
    '''
    	
def read_hyperparameters(cfg, model):
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