# Constant indexes that help reconstruct the graph structure
# Below are the indexes of the node attributes in the x torch vector
import pickle

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_networkx

HEAD_INDEX = 0  # Elevation + base head + initial level
NODE_DIAMETER_INDEX = 1  # Needed for tanks
NODE_TYPE_INDEX = 2
DEMAND_TIMESERIES = slice(3, 27)
# The following three indexes describe edges in the edge_attr torch vector
DIAMETER_INDEX = 0
EDGE_TYPE_INDEX = 1
COEFF_R_INDEX = 2
COEFF_N_INDEX = 3
SCHEDULE_TIMESERIES = slice(4, 28)
# The following three indexes describe the node types inside the x torch vector -> NODE_TYPE_INDEX variable
JUNCTION_TYPE = 0
RESERVOIR_TYPE = 1
TANK_TYPE = 2
# The following two indexes describe the edge types inside the edge_attr torch vector
PIPE_TYPE = 0
PUMP_TYPE = 1



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

def load_raw_dataset_test(wdn_name, data_folder):
    '''
    Load tra/val/data for a water distribution network datasets
    -------
    wdn_name : string
        prefix of pickle files to open
    data_folder : string
        path to datasets
    '''

    data_tst = pickle.load(open(f'{data_folder}/test/{wdn_name}.p', "rb"))

    return data_tst


def create_dataset(database, normalizer=None, output='pressure'):
    '''
    Creates working datasets dataset from the pickle databases
    ------
    database : list
        each element in the list is a pickle file containing Data objects
    normalization: dict
        normalize the dataset using mean and std
    '''

    graphs = []

    for i in database:

        graph = torch_geometric.data.Data()

        # Node attributes

        # type_1H is equal to 1 when the node is a reservoir and 2 when it's a tank

        # The head below is junctions plus reservoirs plus tanks
        head = i.elevation + i.base_head + i.initial_level
        network_characteristics = torch.stack(
            (head, i.node_diameter, i.node_type), dim=1).float()
        demands = i.demand_timeseries
        if demands.dim() == 1:
            total_input = torch.stack((head, i.node_diameter, i.node_type, demands), dim=1).float()
        elif demands.dim() == 2:
            total_input = torch.cat((network_characteristics, demands), dim=1).float()
        graph.x = total_input

        # Position and ID
        graph.pos = i.pos
        graph.ID = i.ID

        # Edge index (Adjacency matrix)
        graph.edge_index = i.edge_index

        # Edge attributes
        # length = i.length
        # roughness = i.roughness
        # schedule = i.schedule
        # graph.edge_attr = torch.stack((diameter, length, roughness), dim=1).float()
        edge_characteristics = torch.stack((i.diameter, i.num_edge_type), dim=1)
        pump_schedules = i.schedule
        pump_coefficients = torch.stack((i.coeff_r, i.coeff_n), dim=1)
        graph.edge_attr = torch.cat((edge_characteristics, pump_coefficients, pump_schedules), dim=1).float()

        ix_edge = graph.edge_index.numpy().T
        ix_edge = (ix_edge[:, 0] < ix_edge[:, 1])
        ix_pump = (graph.edge_attr[:, EDGE_TYPE_INDEX].numpy() == PUMP_TYPE) & ix_edge


        # If the length of the shape of pressure was 2 then it means that the simulation was continuous
        press_shape = i.pressure.shape
        if len(press_shape) == 2:
            graph.y = []
            for time_step in range(press_shape[0]):
                # Appending the tanks to the output since their pressures also need to be predicted like any other node
                nodal_pressures = i.pressure[time_step][[(i.node_type == JUNCTION_TYPE) | (i.node_type == TANK_TYPE)]].reshape(-1, 1)
                selected_flows = i.flowrate[ix_pump]
                pump_flows = selected_flows[:, time_step].reshape(-1, 1)
                output = torch.cat((nodal_pressures, pump_flows), dim=0)
                graph.y.append(output)

        else:
            # Graph output (head)
            if output == 'head':
                graph.y = head[i.node_type == 0].reshape(-1, 1)
            else:
                graph.y = i.pressure[i.node_type == 0].reshape(-1, 1)
        # normalization
        if normalizer is not None:
            graph = normalizer.transform(graph)

        graphs.append(graph)

    A12 = nx.incidence_matrix(to_networkx(graphs[0]), oriented=True).toarray().transpose()
    return graphs, A12


def create_dataset_MLP_from_graphs(graphs, features=['base_heads', 'diameter', 'coeff_r', 'coeff_n',
                                                     'pump_schedules', 'demand_timeseries',], no_res_out=True):
    # index edges to avoid duplicates: this considers all graphs to be UNDIRECTED!
    ix_edge = graphs[0].edge_index.numpy().T
    ix_edge = (ix_edge[:, 0] < ix_edge[:, 1])
    ix_pipe = (graphs[0].edge_attr[:, EDGE_TYPE_INDEX].numpy() == PIPE_TYPE) & ix_edge
    ix_pump = (graphs[0].edge_attr[:, EDGE_TYPE_INDEX].numpy() == PUMP_TYPE) & ix_edge
    # position of reservoirs, and tanks
    # reservoir type is 1, tank is 2
    ix_junct = graphs[0].x[:, NODE_TYPE_INDEX].numpy() == JUNCTION_TYPE
    ix_res = graphs[0].x[:, NODE_TYPE_INDEX].numpy() == RESERVOIR_TYPE
    ix_tank = graphs[0].x[:, NODE_TYPE_INDEX].numpy() == TANK_TYPE
    indices = {}
    prev_feature = None
    for ix_feat, feature in enumerate(features):
        for ix_item, item in enumerate(graphs):
            if feature == 'base_heads':
                # filter below on ix_res or ix_tank
                ix_res_or_tank = np.logical_or(ix_res, ix_tank)
                x_ = item.x[ix_res_or_tank, HEAD_INDEX]
            elif feature == 'diameter':
                x_ = item.edge_attr[ix_pipe, DIAMETER_INDEX]
            elif feature == 'pump_schedules':
                x_ = item.edge_attr[ix_pump, SCHEDULE_TIMESERIES]
            elif feature == 'demand_timeseries':
                x_ = item.x[ix_junct, DEMAND_TIMESERIES]
            elif feature == 'coeff_r':
                x_ = item.edge_attr[ix_pump, COEFF_R_INDEX]
            elif feature == 'coeff_n':
                x_ = item.edge_attr[ix_pump, COEFF_N_INDEX]
            else:
                raise ValueError(f'Feature {feature} not supported.')

            if ix_item == 0:
                x = x_
            else:
                x = torch.cat((x, x_), dim=0)

        if ix_feat == 0:
            X = x.reshape(len(graphs), -1)
        else:
            if feature == "demand_timeseries":
                x = x.transpose(0, 1)
                check = x.reshape(len(graphs), -1)
                X = torch.cat((X, x.reshape(len(graphs), -1)), dim=1)
            else:
                X = torch.cat((X, x.reshape(len(graphs), -1)), dim=1)
        if prev_feature:
            indices[feature] = slice(indices[prev_feature].stop, X.shape[1], 1)
        else:
            indices[feature] = slice(0, X.shape[1], 1)

        prev_feature = feature

    for ix_item, item in enumerate(graphs):
        # remove reservoirs from y as well
        if ix_item == 0:
            if no_res_out:
                if isinstance(item.y, list):
                    y = torch.stack(item.y).unsqueeze(0).expand(1, -1, -1)
                else:
                    y = item.y
            else:
                y = item.y[ix_junct]

        else:
            if no_res_out:
                if isinstance(item.y, list):
                    y = torch.cat((y, torch.stack(item.y).unsqueeze(0).expand(1, -1, -1)), dim=0)
                else:
                    y = torch.cat((y, item.y), dim=0)
            else:
                y = torch.cat((y, item.y[ix_junct]), dim=0)

    # If the shape of y is 1D then it means that the simulation was single period and should be turned to 2D according to the amount of graphs, if 3D then it was continuous
    if len(y.shape) == 1:
        y = y.reshape(len(graphs), -1)
    # SOMEWHERE HERE THE SEQUENCE LENGTH NEEDS TO BE ADJUSTED
    return torch.utils.data.TensorDataset(X, y), X.shape[1], indices


def create_incidence_matrices(graphs, incidence_matrix):
    # position of reservoirs

    ix_junct = graphs[0].x[:, NODE_TYPE_INDEX].numpy() == JUNCTION_TYPE
    ix_edge = graphs[0].edge_index.numpy().T
    ix_edge = (ix_edge[:, 0] < ix_edge[:, 1])
    incidence_matrix = incidence_matrix[ix_edge, :]
    # The A12 incidence matrix is definitely only junctions, but should the A10 include tanks?
    A10 = incidence_matrix[:, ~ix_junct]
    A12 = incidence_matrix[:, ix_junct]
    A12[np.where(A10 == 1), :] *= -1
    A10[np.where(A10 == 1), :] *= -1
    return A10, A12