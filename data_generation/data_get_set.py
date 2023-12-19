import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr
import math
from sklearn.utils import shuffle


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

    train_dataset = Dataset[:round(N_datasets * train_split)]
    val_dataset = Dataset[
                  round(N_datasets * train_split):(round(N_datasets * train_split) + round(N_datasets * val_split))]
    test_dataset = Dataset[(round(N_datasets * train_split) + round(N_datasets * val_split)):]

    return train_dataset, val_dataset, test_dataset


def load_water_network(inp_file):
    '''
	This function loads a water network inputfile (.inp) and returns a WNTR WaterNetworkModel object.
    ------
    inp_file: .inp file
        file with information of wdn
	'''
    return wntr.network.WaterNetworkModel(inp_file)


def get_attribute_all_nodes(wn, attr_str):
    '''
    This function retrieves an attribute (e.g., base demand) from all nodes in the network.

    output: a pandas Series indexed by node_id and containing the attribute as values.
    ------
    wn: WNTR object
    attr_str: str
        name of the selected attribute e.g., 'base_demand', 'elevation'
    '''
    temp = {}

    for id in wn.node_name_list:
        node = wn.get_node(id)
        try:
            attr = getattr(node, attr_str)
        except AttributeError:
            # e.g., tanks/reservoirs have no base demand
            attr = np.nan
        temp[id] = attr

    return pd.Series(temp)

def get_attribute_all_links(wn, attr_str):
    '''
    This function retrieves an attribute (e.g., diameter) from all links in the network.

    output: a pandas Series indexed by link_id and containing the attribute as values.
    ------
    wn: WNTR object
    attr_str: str
        name of the selected attribute e.g., 'diameter', 'length', 'roughness'
    '''
    temp = {}
    for id in wn.link_name_list:
        link = wn.get_link(id)
        try:
            attr = getattr(link, attr_str)
        except AttributeError:
            # e.g., pumps have no roughness
            attr = np.nan
        temp[id] = attr
    return pd.Series(temp)


def get_attribute_from_networks(attr_str, wn_path, wn_list, plot=True, n_cols=5):
    '''
    This function retrieves and plots the distribution of attribute attr_str across all networks in wn_list.
    The attribute can be either from edges or nodes. At most n_cols histograms are displayed per each row.

    Output: a dictionary with the retrieved attributes across all networks.
    ------
    attr_str: str
        name of the selected attribute e.g., 'diameter', 'length', 'base_demand', 'roughness'
    wn_path: str
        path to the network folder location
    wn_list: list
        names of the networks considered
    plot: bool
        if True, plot the distribution of attr_str for all considered networks
    n_cols: int
        number of plots displayed per each row
    '''
    d_attr = {}

    for network in wn_list:
        inp_file = f'{wn_path}/{network}.inp'
        wn = load_water_network(inp_file)
        # check if attr_str is node attribute
        s_attr = get_attribute_all_nodes(wn, attr_str)
        if s_attr.isnull().all() == True:
            # no? is it a link attribute?
            s_attr = get_attribute_all_links(wn, attr_str)
        if s_attr.isnull().all() == True:
            # no? then the attribute doesn't exist
            raise AttributeError(f'Attribute {attr_str} not existing.')
        d_attr[network] = s_attr

    # plot
    if plot == True:
        n_networks = len(wn_list)
        n_cols = min(n_cols, n_networks)
        n_rows = math.ceil(n_networks / n_cols)
        f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        for ax, network in zip(axes.reshape(-1), d_attr.keys()):
            d_attr[network].hist(ax=ax)
            ax.set_title(network)
        # adjust overall figure
        f.suptitle(attr_str, fontsize=18, color='r')
        f.tight_layout(rect=[0, 0.05, 1, 0.95])  # rect takes into account suptitle

    return d_attr


def get_number_of_components_from_network(wn):
    '''
	This function returns a dictionary with number of links and nodes for a WNTR network model.
	The following total counts are returned:
	Nodes: overall, junctions, reservoirs, storage tanks
	Links: overall, pipes, valves, pumps
	'''
    return {'nodes': wn.num_nodes, 'junctions': wn.num_junctions, 'reservoirs': wn.num_reservoirs,
            'tanks': wn.num_tanks,
            'links': wn.num_links, 'pipes': wn.num_pipes, 'valves': wn.num_valves, 'pumps': wn.num_pumps, }


# get number of components for each network
def get_wdn_components(networks, path):
    '''
    Returns a dataframe with number of network components for each wdn in networks
    ------
    networks: list
        list of wdn names
    path: str
        path to the folder with .inp of the networks
    '''
    for ix, network in enumerate(networks):
        inp_file = f'{path}/{network}.inp'
        wn = load_water_network(inp_file)
        d_counts = get_number_of_components_from_network(wn)
        if ix == 0:
            # create dataframe
            df_counts = pd.DataFrame(index=networks, columns=d_counts.keys())
        df_counts.loc[network, :] = d_counts

    return df_counts


# options for each network
def get_wdn_options(networks, path):
    '''
    Returns a dataframe with hydraulic options for each wdn in networks
    ------
    networks: list
        list of wdn names
    path: str
        path to the folder with .inp of the network
    '''
    df_options = pd.DataFrame(index=networks, columns=['headloss', 'inpfile_units'])
    for network in networks:
        inp_file = f'{path}/{network}.inp'
        wn = load_water_network(inp_file)
        df_options.loc[network, :] = (wn.options.hydraulic.headloss, wn.options.hydraulic.inpfile_units)

    return df_options