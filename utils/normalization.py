import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class PowerLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_transform=False, power=4, reverse=True):
        if log_transform:
            self.log_transform = log_transform
            self.power = None
        else:
            self.power = power
            self.log_transform = None
        self.reverse = reverse
        self.max_ = None
        self.min_ = None

    def fit(self, X, y=None):
        self.max_ = np.max(X)
        self.min_ = np.min(X)
        return self

    def transform(self, X):
        if self.log_transform:
            if self.reverse:
                return np.log1p(self.max_ - X)
            else:
                return np.log1p(X - self.min_)
        else:
            if self.reverse:
                return (self.max_ - X) ** (1 / self.power)
            else:
                return (X - self.min_) ** (1 / self.power)

    def inverse_transform(self, X):
        if self.log_transform:
            if self.reverse:
                return self.max_ - np.exp(X) + 1
            else:
                return np.exp(X) + self.min_ - 1
        else:
            if self.reverse:
                return self.max_ - X ** self.power
            else:
                return X ** self.power + self.min_


class GraphNormalizer:
    def __init__(self, junct_and_tanks=None, x_feat_names=['head', 'diameter', 'type', 'demand_timeseries'],
                 ea_feat_names=['diameter'], output='pressure'):
        # store
        self.x_feat_names = x_feat_names
        self.ea_feat_names = ea_feat_names
        self.output = output
        self.junct_and_tanks = junct_and_tanks

        # create separate scaler for each feature (can be improved, e.g., you can fit a scaler for multiple columns)
        self.scalers = {}
        for feat in self.x_feat_names:
            if feat == 'head':
                self.scalers[feat] = PowerLogTransformer(log_transform=True, reverse=False)
            else:
                self.scalers[feat] = MinMaxScaler()

        for feat in self.ea_feat_names:
            if feat == 'length':
                self.scalers[feat] = PowerLogTransformer(log_transform=True, reverse=False)
            else:
                self.scalers[feat] = MinMaxScaler()

        if isinstance(self.output, list):
            for element in self.output:
                if element == 'pressure':
                    self.scalers[element] = PowerLogTransformer(log_transform=True, reverse=True)
                else:
                    self.scalers[element] = MinMaxScaler()
        else:
            self.scalers[output] = PowerLogTransformer(log_transform=True, reverse=True)

    def fit(self, graphs):
        ''' Fit the scalers on an array of x and ea features
        '''
        x, y, ea = from_graphs_to_pandas(graphs)

        for ix, feat in enumerate(self.x_feat_names):
            self.scalers[feat] = self.scalers[feat].fit(x[:, ix].reshape(-1, 1))

        for ix, feat in enumerate(self.ea_feat_names):
            self.scalers[feat] = self.scalers[feat].fit(ea[:, ix].reshape(-1, 1))

        if isinstance(self.output, list):
            for element in self.output:
                if element == 'pressure':
                    self.scalers[element] = self.scalers[element].fit(y[:, :self.junct_and_tanks].reshape(-1, 1))
                else:
                    if len(y[:, self.junct_and_tanks:].reshape(-1, 1)) > 0:
                        # What is beyond the junct_and_tanks is the pump flow if there are pumps
                        self.scalers[element] = self.scalers[element].fit(y[:, self.junct_and_tanks:].reshape(-1, 1))
        else:
            self.scalers[self.output] = self.scalers[self.output].fit(y.reshape(-1, 1))

        return self

    def transform(self, graph):
        ''' Transform graph based on normalizer
        '''
        graph = graph.clone()
        for ix, feat in enumerate(self.x_feat_names):
            if feat != 'type' and feat != 'pump_schedules':
                temp = graph.x[:, ix].numpy().reshape(-1, 1)
                graph.x[:, ix] = torch.tensor(self.scalers[feat].transform(temp).reshape(-1))
        for ix, feat in enumerate(self.ea_feat_names):
            temp = graph.edge_attr[:, ix].numpy().reshape(-1, 1)
            graph.edge_attr[:, ix] = torch.tensor(self.scalers[feat].transform(temp).reshape(-1))

        if isinstance(graph.y, list):
            transformed_y = []
            for i, el in enumerate(graph.y):
                if isinstance(self.output, list):
                    for element in self.output:
                        if element == 'pressure':
                            pressure = torch.tensor(self.scalers[element].transform(graph.y[i][:self.junct_and_tanks].numpy().reshape(-1, 1)).reshape(-1))
                        else:
                            if len(graph.y[i][self.junct_and_tanks:, :].reshape(-1, 1)) > 0:
                                pump_flow = torch.tensor(self.scalers[element].transform(graph.y[i][self.junct_and_tanks:].numpy().reshape(-1, 1)).reshape(-1))
                            else:
                                pump_flow = None
                    if pump_flow is not None:
                        transformed_y.append(torch.cat((pressure, pump_flow), dim=0))
                    else:
                        transformed_y.append(pressure)
                else:
                    transformed_y.append(
                        torch.tensor(self.scalers[self.output].transform(graph.y[i].numpy().reshape(-1, 1)).reshape(-1)))
            graph.y = transformed_y
        else:
            graph.y = torch.tensor(self.scalers[self.output].transform(graph.y.numpy().reshape(-1, 1)).reshape(-1))
        return graph

    # def inverse_transform(self, graph):
    #     ''' Perform inverse transformation to return original features
    #     '''
    #     graph = graph.clone()
    #     for ix, feat in enumerate(self.x_feat_names):
    #         temp = graph.x[:, ix].numpy().reshape(-1, 1)
    #         graph.x[:, ix] = torch.tensor(self.scalers[feat].inverse_transform(temp).reshape(-1))
    #     for ix, feat in enumerate(self.ea_feat_names):
    #         temp = graph.edge_attr[:, ix].numpy().reshape(-1, 1)
    #         graph.edge_attr[:, ix] = torch.tensor(self.scalers[feat].inverse_transform(temp).reshape(-1))
    #     graph.y = torch.tensor(self.scalers[self.output].inverse_transform(graph.y.numpy().reshape(-1, 1)).reshape(-1))
    #     return graph

    def transform_array(self, z, feat_name):
        '''
            This is for MLP dataset; it can be done better (the entire thing, from raw data to datasets)
        '''
        return torch.tensor(self.scalers[feat_name].transform(z).reshape(-1))

    def inverse_transform_array(self, z, feat_name, reshape=True):
        '''
            This is for MLP dataset; it can be done better (the entire thing, from raw data to datasets)
        '''

        if reshape:
            return torch.tensor(self.scalers[feat_name].inverse_transform(z).reshape(-1))
        else:
            return torch.tensor(self.scalers[feat_name].inverse_transform(z))

    def denormalize_multiple(self, array, output_nodes):

        # reshape to select correct ones and then reshape before transform
        array_reshaped = array.reshape(-1, output_nodes)
        array_only_pressures = array_reshaped[:, :self.junct_and_tanks].reshape(-1, 1)
        array_only_flows = array_reshaped[:, self.junct_and_tanks:].reshape(-1, 1)
        array_pressures = self.inverse_transform_array(array_only_pressures, 'pressure')

        if len(array_only_flows) > 0:
            array_flow = self.inverse_transform_array(array_only_flows, 'pump_flow')

            pred = torch.cat((array_pressures.reshape(-1, (self.junct_and_tanks)),
                          array_flow.reshape(-1, output_nodes - (self.junct_and_tanks))), axis=1)
        else:
            pred = array_pressures.reshape(-1, (self.junct_and_tanks))

        return pred
def from_graphs_to_pandas(graphs):
    x = []
    y = []
    ea = []
    for i, graph in enumerate(graphs):
        x.append(graph.x.numpy())

        if isinstance(graph.y, list):

            y.append(np.stack(graph.y, axis=0))
        else:
            y.append(graph.y.numpy())

        ea.append(graph.edge_attr.numpy())

    return np.concatenate(x, axis=0), np.concatenate(y, axis=0), np.concatenate(ea, axis=0)