import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wntr
import networkx as nx
from tqdm import tqdm
import math
from scipy import stats
import torch
from torch_geometric.utils import convert
import os
from sklearn.utils import shuffle
from pathlib import Path
import time
from scipy.optimize import curve_fit

import demand_generation


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

def generate_binary_string(length=24, zero_probability=0.05):
    binary_schedule = []
    for _ in range(length):
        if np.random.random() < zero_probability:
            binary_schedule.append(0)
        else:
            binary_schedule.append(1)
    return binary_schedule

def run_wntr_simulation(wn, headloss='H-W', continuous=False):
    '''
	This function runs a simulation after changing the hydraulic options.
    ------
    wn: WNTR object
    headloss: str
        options: 'H-W'= Hazen-Williams, 'D-W'=Darcy-Weisbach
	'''
    wn.options.hydraulic.viscosity = 1.0
    wn.options.hydraulic.specific_gravity = 1.0
    wn.options.hydraulic.demand_multiplier = 1.0
    wn.options.hydraulic.demand_model = 'DD'
    wn.options.hydraulic.minimum_pressure = 0
    wn.options.hydraulic.required_pressure = 1
    wn.options.hydraulic.pressure_exponent = 0.5
    wn.options.hydraulic.headloss = headloss
    wn.options.hydraulic.trials = 50
    wn.options.hydraulic.accuracy = 0.001
    wn.options.hydraulic.unbalanced = 'CONTINUE'
    wn.options.hydraulic.unbalanced_value = 10
    wn.options.hydraulic.checkfreq = 2
    wn.options.hydraulic.maxcheck = 10
    wn.options.hydraulic.damplimit = 0.0
    wn.options.hydraulic.headerror = 0.0
    wn.options.hydraulic.flowchange = 0.0
    wn.options.hydraulic.inpfile_units = "LPS"

    if continuous:
        wn.options.time.duration = 82800  # 24 * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.quality_timestep = 3600
        wn.options.time.report_start = 0
        wn.options.time.report_timestep = 3600
        wn.options.time.pattern_start = 0
        wn.options.time.pattern_timestep = 3600
    else:
        wn.options.time.duration = 0

    start_time = time.time()
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(version=2.2)

    end_time = time.time()

    print(f"Simulation time: {end_time - start_time}")
    return results


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


def alter_water_network(wn, continuous, randomized_demands=None):
    '''
	This function randomly modifies nodes and edges attributes in the water network according to the distributions in d_attr.
	At the moment, these are expressed as arrays containing all possible values. No changes are made if d_attr=None.
	No changes are made to a particular attribute if it is not in the keys of d_attr.
	'''
    set_attribute_all_nodes_rand(wn, continuous, randomized_demands)
    set_attribute_all_pumps_rand(wn, continuous)
    return None

def set_attribute_all_pumps_rand(wn, continuous):
    """
	This function changes an attribute attr_str (e.g., roughness) from all links in the network based on their orignal value.
	The list of potential values is contained in attr_values. search_range identifies how many values to the left and to the right
	of the original value are considered for the random selection.

	Tested for: diameter, roughness
	"""
    # IN_TO_M = 0.0254

    # if attr_str not in ['diameter', 'roughness']:
    #     raise AttributeError('You can only change pipe roughness and diameter as link attributes.')

    for id in wn.link_name_list:
        link = wn.get_link(id)
        if link.link_type == 'Pump':
            pattern = wn.get_pattern(link.speed_pattern_name)
            pattern.multipliers = generate_binary_string()

    return None

def set_attribute_all_nodes_rand(wn, continuous, randomized_demands):
    '''
	This function changes an attribute attr_str (e.g., base demand) from all nodes in the network based on their orignal value.
	The list of potential values is contained in attr_values. search_range identifies how many values to the left and to the right
	of the original value are considered for the random selection.

	Tested for: demand_multipliers
	'''

    # Setting the demand per node to a random value out of three randomly generated types of households
    if continuous:
        for i in randomized_demands:
            wn.add_pattern(f'demand_pattern_{i}',
                           randomized_demands[i])

    total_demands = []

    for id in wn.nodes.junction_names:
        node = wn.get_node(id)
        node.demand_timeseries_list[0].base_value = np.random.choice(range(0, 1000)) * 0.000002
        base_val = node.demand_timeseries_list[0].base_value
        # np.random.choice([0.0000008, 0.0000001, 0.00000002]))
        if continuous:
            node.demand_timeseries_list[0].pattern_name = 'demand_pattern_{}'.format(
                np.random.choice(['one_person', 'two_person', 'family']))
            this_demand = base_val * wn.get_pattern(node.demand_timeseries_list[0].pattern_name).multipliers

        this_demand = this_demand * 3600
        total_demands.append(sum(this_demand))

    print(sum(total_demands))

    return None



def set_attribute_all_links_rand(wn, attr_str, attr_values, search_range, prob_exp=0):
    """
	This function changes an attribute attr_str (e.g., roughness) from all links in the network based on their orignal value.
	The list of potential values is contained in attr_values. search_range identifies how many values to the left and to the right
	of the original value are considered for the random selection.

	Tested for: diameter, roughness
	"""
    # IN_TO_M = 0.0254

    if attr_str not in ['diameter', 'roughness']:
        raise AttributeError('You can only change pipe roughness and diameter as link attributes.')
    for id in wn.link_name_list:
        link = wn.get_link(id)
        if hasattr(link, attr_str):
            attr = getattr(link, attr_str)
            setattr(link, attr_str, select_random_value(attr, attr_values, search_range, prob_exp))
    return None


def select_random_value(current_value, possible_values, search_range, prob_exp=0):
    '''
	This function randomly selects a value in a list, centered on the current value.

	prob_exp is the exponent (0=uniform, 1=linear, 2=quadratic,...) use to nudge np.random.choice selection
	towards higher possible_values (prob_exp >= 1); this is useful for diameter (to obtain pressure distribution closer to that of original map).
	'''
    if search_range == 0:
        return (current_value)

    if all(possible_values[i] < possible_values[i + 1] for i in range(len(possible_values) - 1)) == False:
        raise ValueError('List of possible values is not sorted or contains duplicates.')

    if current_value >= possible_values[-1]:
        max_pos = len(possible_values)
        min_pos = max(max_pos - search_range, 0)
    else:
        max_pos = min((np.array(possible_values) > current_value).argmax() + search_range, len(possible_values))
        min_pos = max((np.array(possible_values) > current_value).argmax() - search_range - 1, 0)

    # assign probabilities
    if prob_exp > 0:
        len_seg = max_pos - min_pos
        prob = np.arange(1, len_seg + 1) ** prob_exp / sum(np.arange(1, len_seg + 1) ** prob_exp)
        return np.random.choice(possible_values[min_pos:max_pos], p=prob)

    return np.random.choice(possible_values[min_pos:max_pos])


def get_dataset_entry(network, path, continuous=False, randomized_demands=None):
    '''
	This function creates a random input/output pair for a single network, after modifying it the original wds model.
	'''
    link_feats = ['diameter']
    node_feats = ['base_demand', 'node_type', 'base_head']
    res_dict = {}
    # load and alter network
    inp_file = f'{path}/{network}.inp'
    wn = load_water_network(inp_file)

    alter_water_network(wn, continuous, randomized_demands)
    # retrieve input features
    for feat in link_feats:
        res_dict[feat] = get_attribute_all_links(wn, feat)
    for feat in node_feats:
        res_dict[feat] = get_attribute_all_nodes(wn, feat)
    # get output == pressure, after running simulation
    sim = run_wntr_simulation(wn, headloss='H-W', continuous=continuous)
    res_dict['pressure'] = sim.node['pressure'].squeeze()
    res_dict['flowrate'] = sim.link['flowrate'].squeeze()
    # check simulation
    ix = res_dict['node_type'][res_dict['node_type'] == 'Junction'].index.to_list()
    sim_check = ((res_dict['pressure'][ix] > 1).all()) & (sim.error_code == None)
    res_dict['network_name'] = network
    res_dict['network'] = wn
    return res_dict, sim, sim_check


def create_dataset(network, path, n_trials, max_fails=1e6, continuous=False, randomized_demands=None, count=2):
    """
    This function creates a dataset of n_trials length for a specific network
    """
    n_fails = 0
    dataset = []

    for i in tqdm(range(n_trials), network):
        flag = False
        while not flag:
            if i != 0 and i % 1000 == 0:
                randomized_demands = demand_generation.generate_demand_patterns()
            res_dict, _, flag = get_dataset_entry(network, path, continuous, randomized_demands=randomized_demands)
            # The flag below is used to check if the simulation is correct
            # It is one boolean for steady state but a list for the continuous options

            # Pandas series "flag" to list and if there is one False make the whole list false
            if isinstance(flag, pd.Series):
                flag = flag.tolist()
                if False in flag:
                    flag = False
                else:
                    if count > 0:
                        count -= 1
                        wntr.network.write_inpfile(res_dict['network'], f'{path}/{network}_{count}.inp')
                    flag = True

            if not flag:
                n_fails += 1
            if n_fails >= max_fails:
                raise RecursionError(f'Max number of fails ({max_fails}) reached.')
        dataset.append(res_dict)
    print("Total number of fails was ", n_fails)
    return dataset


def plot_dataset_attribute_distribution(dataset, attribute, figsize=(20, 5), bins=20):
    '''
    This function plots the overall distribution of a dataset property (e.g., nodes).
    Different colors are used for different networks within the dataset.
    '''
    df = pd.DataFrame(dataset)
    networks = df.network
    df_attr = pd.DataFrame()
    for network in networks.unique().tolist():
        net_attr = []
        for row in df[df.network == network][attribute]:
            net_attr += row.values.tolist()
            df_attr = pd.concat([df_attr, pd.Series(net_attr)], axis=1)
    df_attr.columns = networks

    # get limits
    min_x = df_attr.min().min()
    max_x = df_attr.max().max()

    # plot
    axes = df_attr.head(100).hist(figsize=figsize, bins=bins)
    for ax in axes.flatten():
        ax.set_xlim([min_x, max_x])
        plt.tight_layout()

    return df_attr


def from_wntr_to_nx(wn, continuous):
    '''
	This function converts a WNTR object to networkx
	'''
    wn_links = list(wn.links())
    wn_nodes = list(wn.nodes())

    G_WDS = wn.get_graph()  # directed multigraph
    uG_WDS = G_WDS.to_undirected()  # undirected
    sG_WDS = nx.Graph(uG_WDS)  # Simple graph

    for (u, v, wt) in sG_WDS.edges.data():
        for edge in wn.links():
            if (edge[1].start_node.name == u and edge[1].end_node.name == v) or (
                    edge[1].start_node.name == v and edge[1].end_node.name == u):

                if sG_WDS[u][v]['type'] == 'Pipe':
                    sG_WDS[u][v]['name'] = edge[1].name
                    sG_WDS[u][v]['diameter'] = edge[1].diameter
                    # sG_WDS[u][v]['length'] = edge[1].length
                    # sG_WDS[u][v]['roughness'] = edge[1].roughness
                    sG_WDS[u][v]['num_edge_type'] = 0
                    sG_WDS[u][v]['coeff_r'] = 0
                    sG_WDS[u][v]['coeff_n'] = 0
                    sG_WDS[u][v]['schedule'] = torch.tensor(np.array([0] * 24))

                elif sG_WDS[u][v]['type'] == 'Pump':
                    # Only handling Power Pumps for now
                    sG_WDS[u][v]['name'] = edge[1].name
                    sG_WDS[u][v]['diameter'] = 0
                    # sG_WDS[u][v]['length'] = 0
                    # sG_WDS[u][v]['roughness'] = 0
                    sG_WDS[u][v]['num_edge_type'] = 1
                    curve = wn.get_curve(edge[1].pump_curve_name)
                    curve_points = curve.points

                    # If there's only one point, assume a single-point curve as described
                    if len(curve_points) == 1:
                        head, flow = curve_points[0]
                        # Correcting EPANET bug
                        head = head * 1000
                        shutoff_head = 1.33 * head
                        max_flow = 2 * flow

                        # Treat it as a three-point curve
                        curve_points = [(0, shutoff_head), (flow, head), (max_flow, 0)]
                    q_data = np.array([x[0] for x in curve_points])
                    h_data = np.array([x[1] for x in curve_points])

                    def pump_curve(q, A, B, C):
                        return A - B * np.power(q, C)

                    popt, _ = curve_fit(pump_curve, q_data, h_data, maxfev=10000)
                    A, B, C = popt
                    sG_WDS[u][v]['coeff_r'] = B  # Coeff R = B
                    sG_WDS[u][v]['coeff_n'] = C  # Coeff N = C

                    speed_pattern = wn.get_pattern(edge[1].speed_pattern_name).multipliers
                    if speed_pattern is None:
                        sG_WDS[u][v]['schedule'] = [0] * 24
                    else:
                        sG_WDS[u][v]['schedule'] = torch.tensor(speed_pattern)

                else:
                    print(sG_WDS[u][v]['type'], u, v)
                    raise Exception('Only Pipes and Pumps so far')
                    break

    i = 0
    for u in sG_WDS.nodes:
        # Junctions have elevation but no base_head and are identified with a 0
        if sG_WDS.nodes[u]['type'] == 'Junction':
            sG_WDS.nodes[u]['ID'] = wn_nodes[i][1].name
            sG_WDS.nodes[u]['node_type'] = 0
            sG_WDS.nodes[u]['elevation'] = wn_nodes[i][1].elevation
            sG_WDS.nodes[u]['base_head'] = 0
            sG_WDS.nodes[u]['initial_level'] = 0
            sG_WDS.nodes[u]['node_diameter'] = 0

            if continuous:
                pattern_name = wn_nodes[i][1].demand_timeseries_list.to_list()[0]['pattern_name']

                multipliers = wn.get_pattern(pattern_name).multipliers
                # Not sure about the multiplicatiobn below. Shouldn't really matter anyway
                value = wn_nodes[i][1].demand_timeseries_list[0].base_value * 1000
                mul_val = multipliers * value
                sG_WDS.nodes[u]['demand_timeseries'] = torch.tensor(mul_val)
            else:
                sG_WDS.nodes[u]['demand_timeseries'] = wn_nodes[i][1].base_demand

        # Reservoirs have base_head but no elevation and are identified with a 1
        elif sG_WDS.nodes[u]['type'] == 'Reservoir':
            sG_WDS.nodes[u]['ID'] = wn_nodes[i][1].name
            sG_WDS.nodes[u]['node_type'] = 1
            sG_WDS.nodes[u]['elevation'] = 0
            sG_WDS.nodes[u]['base_head'] = wn_nodes[i][1].base_head
            sG_WDS.nodes[u]['initial_level'] = 0
            sG_WDS.nodes[u]['node_diameter'] = 0

            if continuous:
                sG_WDS.nodes[u]['demand_timeseries'] = torch.tensor(np.array([0] * 24))
            else:
                sG_WDS.nodes[u]['demand_timeseries'] = 0

        # Tanks have an elevation, as well as initial_level and diameter
        elif sG_WDS.nodes[u]['type'] == 'Tank':
            sG_WDS.nodes[u]['ID'] = wn_nodes[i][1].name
            sG_WDS.nodes[u]['node_type'] = 2
            sG_WDS.nodes[u]['elevation'] = wn_nodes[i][1].elevation
            sG_WDS.nodes[u]['base_head'] = 0
            sG_WDS.nodes[u]['initial_level'] = wn_nodes[i][1].init_level
            sG_WDS.nodes[u]['node_diameter'] = wn_nodes[i][1].diameter
            if continuous:
                sG_WDS.nodes[u]['demand_timeseries'] = torch.tensor(np.array([0] * 24))
            else:
                sG_WDS.nodes[u]['demand_timeseries'] = 0

        else:
            print(u)
            raise Exception('Only Junctions, Reservoirs and Tanks so far')
            break
        i += 1

    return sG_WDS  # df_nodes, df_links, sG_WDS


def convert_to_pyg(dataset, continuous):
    '''
    This function converts a list of simulations into a PyTorch Geometric Data type
    ------
    dataset: list
        list of network simulations, as given by create_dataset
    '''
    all_pyg_data = []

    for sample in dataset:
        wn = sample['network']
        flows = sample['flowrate']
        # create PyG Data
        pyg_data = convert.from_networkx(from_wntr_to_nx(wn, continuous))

        # Add network name
        pyg_data.name = sample['network_name']
        # Add diameters for MLP
        pyg_data.diameters = torch.tensor(sample['diameter']).float()
        # Add simulation results
        if continuous:
            press_shape = sample['pressure'].shape
            press_reshaped = sample['pressure'].values.reshape(press_shape[0], press_shape[1])
            pyg_data.pressure = torch.tensor(press_reshaped)
        else:
            pyg_data.pressure = torch.tensor(sample['pressure'])

        # convert to float where needed
        pyg_data.diameter = pyg_data.diameter.float()

        all_pyg_data.append(pyg_data)

    return all_pyg_data


def save_database(database, names, size, out_path):
    '''
    This function saves the geometric database into a pickle file
    The name of the file is given by the used networks and the number of simulations
    ------
    database: list
        list of geometric datasets
    names: list
        list of the network names, possibly ordered by number of nodes
    size: int
        number of simulations per each network
    out_path: str
        output file location
    '''
    if isinstance(names, list):
        name = names + [str(size)]
        name = '_'.join(name)
    elif isinstance(names, str):
        name = names  # + '_' + str(size)

    Path(out_path).mkdir(parents=True, exist_ok=True)

    # No need to dump the whole dataset since we generate the sets
    # pickle.dump(database, open(f"{out_path}\\{name}.p", "wb"))

    train_dataset, val_dataset, test_dataset = train_val_test(database)

    pickle.dump(train_dataset, open(f"{out_path}train\\{name}.p", "wb"))
    pickle.dump(val_dataset, open(f"{out_path}valid\\{name}.p", "wb"))
    pickle.dump(test_dataset, open(f"{out_path}test\\{name}.p", "wb"))

    return None


def create_and_save(network, net_path, n_trials, out_path, max_fails=1e4, continuous=False, randomized_demands=None):
    '''
    Creates and saves dataset given a list of networks and possible range of variable variations
    ------
    networks: list or str
        list or string of wdn names
    net_path: str
        path to the folder with .inp of the networks
    n_trials: int
        number of simulations
    d_attr: dict
        dictionary with values for each attribute
    d_newt: dict
        dictionary with ranges for each network
    out_path: str
        output file location
    max_fails: int
        number of maximum failed simulations per network
    show: bool
        if True, shows a bar progression for each simulation
    continuous: bool
        if True, the simulation is run for 24 hours instead of a single period
    '''
    # create dataset
    all_data = []

    start_time = time.time()
    all_data += create_dataset(network, net_path, n_trials, max_fails=max_fails, continuous=continuous,
                               randomized_demands=randomized_demands, count=0)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds\n")

    # Create PyTorch Geometric dataset
    all_pyg_data = convert_to_pyg(all_data, continuous)

    # Save database
    save_database(all_pyg_data, names=network, size=n_trials, out_path=out_path)

    return None


def import_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''
    with open(config_file) as f:
        data = yaml.safe_load(f)
        networks = data['dataset_names']
        n_trials = data['n_trials']

    return networks, n_trials
