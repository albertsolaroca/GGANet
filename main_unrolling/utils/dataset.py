import pickle
import torch
import numpy as np
from itertools import product
import torch_geometric
from sklearn.utils import shuffle

from models.virtual_nodes import add_virtual_nodes
from utils.scaling import *


# constant indexes for node and edge features
ELEVATION_INDEX = 0
BASEDEMAND_INDEX = 1
BASEHEAD_INDEX = 2
DIAMETER_INDEX = 0
LENGTH_INDEX = 1
ROUGHNESS_INDEX = 2

def load_raw_dataset(wdn_name, data_folder):
    '''
    Load tra/val/data for a water distribution network datasets
    -------
    wdn_name : string
        prefix of pickle files to open
    data_folder : string
        path to datasets
    '''

    data_tra = pickle.load(open(f'{data_folder}/train/{wdn_name}.p', "rb"))
    data_val = pickle.load(open(f'{data_folder}/valid/{wdn_name}.p', "rb"))
    data_tst = pickle.load(open(f'{data_folder}/test/{wdn_name}.p', "rb"))

    return data_tra, data_val, data_tst

def create_dataset(database, normalizer=None, HW_rough_minmax=[60, 150],add_virtual_reservoirs=False, output='pressure'):
    '''
    Creates working datasets dataset from the pickle databases	
    ------    
    database : list
        each element in the list is a pickle file containing Data objects
    normalization: dict
        normalize the dataset using mean and std
    '''
    # Roughness info (Hazen-Williams) / TODO: remove the hard_coding
    minR = HW_rough_minmax[0]
    maxR = HW_rough_minmax[1]

    graphs = []

    for i in database:
        graph = torch_geometric.data.Data()

        # Node attributes
        # elevation_head = i.elevation + i.base_head
        # elevation_head = i.elevation.clone()
        # elevation_head[elevation_head == 0] = elevation_head.mean()

        min_elevation = min(i.elevation[i.type_1H == 0])
        head = i.pressure + i.base_head + i.elevation                
        # elevation_head[i.type_1H == 1] = head[i.type_1H == 1]
        # elevation = elevation_head - min_elevation

        # base_demand = i.base_demand * 1000  # convert to l/s        
        # graph.x = torch.stack((i.elevation, i.base_demand, i.type_1H*i.base_head), dim=1).float()
        graph.x = torch.stack((i.elevation+i.base_head, i.base_demand, i.type_1H), dim=1).float()        
        # graph.x = torch.stack((i.elevation+i.base_head, i.base_demand, i.type_1H), dim=1).float()

        # Position and ID
        # graph.pos = i.pos
        graph.ID = i.ID

        # Edge index (Adjacency matrix)
        graph.edge_index = i.edge_index

        # Edge attributes
        diameter = i.diameter
        length = i.length
        roughness = i.roughness
        graph.edge_attr = torch.stack((diameter, length, roughness), dim=1).float()

        # pressure = i.pressure
        # graph.y = pressure.reshape(-1,1)

        # Graph output (head)
        if output == 'head':
            raise ValueError('Not yet implemented')
            # head = head.reshape(-1, 1)
            # head[i.type_1H == 1] = -torch.nan
            # graph.y = head
        else:
            pressure = i.pressure.reshape(-1, 1)
            pressure[i.type_1H == 1] = 0 # THIS HAS TO BE DONE BETTER
            graph.y = pressure
            
        
        # normalization
        if normalizer is not None:
            graph = normalizer.transform(graph)

        if add_virtual_reservoirs:

            graph.x = torch.nn.functional.pad(graph.x, (0, 1))
            graph.edge_attr = torch.nn.functional.pad(graph.edge_attr, (0, 1))
            add_virtual_nodes(graph)
            
        graphs.append(graph)
    return graphs


def create_dataset_MLP(database, normalizer=None, features=['diameter', 'base_demand', 'roughness']):
    '''
    TO DO
    '''

    # index edges to avoid duplicates: this considers all graphs to be UNDIRECTED!
    ix_edge = database[0].edge_index.numpy().T
    ix_edge = (ix_edge[:, 0] < ix_edge[:, 1])

    for ix_feat, feature in enumerate(features):
        for ix_item, item in enumerate(database):
            x_ = getattr(item, feature)
            if feature in ['diameter', 'roughness']:
                # check needed to avoid duplicates
                x_ = x_[ix_edge]
            elif feature not in ['base_demand']:
                raise ValueError(f'Feature {feature} not supported.')
            if ix_item == 0:
                x = x_
            else:
                x = torch.cat((x, x_), dim=0)
        if ix_feat == 0:
            X = x.reshape(len(database), -1)
        else:
            X = torch.cat((X, x.reshape(len(database), -1)), dim=1)

    for ix_item, item in enumerate(database):
        if ix_item == 0:
            y = item.pressure
        else:
            y = torch.cat((y, item.pressure), dim=0)
    y = y.reshape(len(database), -1)

    if normalizer is not None:
        pressure_max = normalization['pressure']
        y = y / pressure_max

    return torch.utils.data.TensorDataset(X, y)

def create_dataset_MLP_from_graphs(graphs, features=['diameter', 'base_demand', 'roughness']):
    '''
    TO DO
    '''

    # index edges to avoid duplicates: this considers all graphs to be UNDIRECTED!
    ix_edge = graphs[0].edge_index.numpy().T
    ix_edge = (ix_edge[:, 0] < ix_edge[:, 1])

    for ix_feat, feature in enumerate(features):
        for ix_item, item in enumerate(graphs):
            if feature == 'diameter':                
                x_ = item.edge_attr[ix_edge,DIAMETER_INDEX]
            elif feature == 'roughness':
                x_ = item.edge_attr[ix_edge,ROUGHNESS_INDEX]
            elif feature == 'base_demand':
                x_ = item.x[:,BASEDEMAND_INDEX]
            else:
                raise ValueError(f'Feature {feature} not supported.')
            if ix_item == 0:
                x = x_
            else:
                x = torch.cat((x, x_), dim=0)
        if ix_feat == 0:
            X = x.reshape(len(graphs), -1)
        else:
            X = torch.cat((X, x.reshape(len(graphs), -1)), dim=1)

    for ix_item, item in enumerate(graphs):
        if ix_item == 0:
            y = item.y
        else:
            y = torch.cat((y, item.y), dim=0)
    y = y.reshape(len(graphs), -1)

    return torch.utils.data.TensorDataset(X, y)    