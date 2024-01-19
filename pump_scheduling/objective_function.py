# Imports
import numpy as np
import pandas as pd
import torch

from get_set import *
from main_unrolling.tune_train import prepare_scheduling
import metamodel as metamodel


# Changes the pump schedule values (as multipliers from 0 to 1) for a given pump
def change_pumping_pattern(wn, pump_pattern_id, new_pattern_values):
    for i in range(len(pump_pattern_id)):
        pattern = wn.get_pattern(pump_pattern_id[i])
        pattern.multipliers = new_pattern_values[i]

    return


# Changes the demand pattern values (as multipliers from 0 to 1) for a given pattern
def change_demand_pattern(wn, demand_pattern_id, new_pattern_values):
    pattern = wn.get_pattern(demand_pattern_id)
    pattern.multipliers = new_pattern_values
    return


# Returns the energy consumption of a given pump in kWh
def energy_consumption(wn, pump_flowrate, head):
    # Energy consumption in kWh
    energy = wntr.metrics.pump_energy(pump_flowrate, head, wn)  # in J
    energy = energy / 3600000  # in kWh

    return energy


# Returns the energy cost of a given pump per timestep
def energy_cost(energy, wn):
    cost = wntr.metrics.pump_cost(energy, wn)
    return cost


# Returns the total energy consumption and cost for a given pump
def total_energy_and_cost(wn, pump_flowrate, head, pump_id_list, timestep=3600):
    energy = energy_consumption(wn, pump_flowrate, head)
    cost = energy_cost(energy, wn)
    total_energy_per_pump = []
    total_cost_per_pump = []

    for pump in pump_id_list:
        # wn.get_link(pump).energy_price = 0.75
        # Energy consumption & Cost
        pump_energy_series = []
        pump_cost_series = []

        for time_loc in range(0, timestep * 24, timestep):
            # Energy in Watts divided by 1000 to get kW consumed per hour (kWh)
            pump_energy_series.append(energy.loc[time_loc, pump])
            pump_cost_series.append(cost.loc[time_loc, pump])

        total_energy_per_pump.append(sum(pump_energy_series))
        total_cost_per_pump.append(sum(pump_cost_series))

    total_energy = sum(total_energy_per_pump)
    total_cost = sum(total_cost_per_pump)

    return total_energy, total_cost


# Calculates total energy consumption for an input of new pumping patterns list, \
# and list of pump_ids, list of critical nodes and demand pattern id
def calculate_objective_function(wn, result):

    pump_id_list = wn.pump_name_list
    pump_flowrate = result.link['flowrate'].loc[:, wn.pump_name_list]
    # Heads
    head = result.node['head']

    calculation = total_energy_and_cost(wn, pump_flowrate, head, pump_id_list)
    total_energy = calculation[0]
    total_cost = calculation[1]

    critical_node_pressures = []

    for node in get_junction_nodes(wn):
        pressure = min(get_pressure_at_node(result, node))
        critical_node_pressures.append(pressure)

    for node in get_tank_nodes(wn):
        pressure = min(get_head_at_node(result, node))
        critical_node_pressures.append(pressure)

    return total_energy, total_cost, critical_node_pressures


# Calculates total energy consumption for an input of new pumping patterns list, \
# and list of pump_ids, list of critical nodes and demand pattern id
def calculate_objective_function_mm(network_file, energy_price, result, node_idx, names, timestep=3600):
    wn = make_network('../data_generation/networks/' + network_file + '.inp')
    # get wn patterns
    pats = wn.patterns

    wn.add_pattern('EnergyPrice', energy_price)
    wn.options.energy.global_pattern = 'EnergyPrice'

    pump_id_list = wn.pump_name_list
    node_ids = np.array(names['node_ids'], copy=False)
    edge_ids = np.array(names['edge_ids'], copy=False)
    edge_types = np.array(names['edge_types'], copy=False)
    node_types = np.array(names['node_types'], copy=False)

    jt_filter = np.where((node_types == 'Junction') | (node_types == 'Tank'))[0]
    jt_ids = node_ids[jt_filter]

    pump_filter = np.where(edge_types == 'Pump')[0]
    pump_ids = set(edge_ids[pump_filter])

    pump_flowrate_raw = result[:, node_idx:] / 1000
    pressures_raw = result[:, :node_idx]

    # Heads
    pressures = pd.DataFrame(data=pressures_raw, index=range(0, timestep * 24, timestep), columns=jt_ids)
    heads = pd.DataFrame(data=names['node_heads'], index=range(0, timestep * 24, timestep))

    total_heads = heads.add(pressures, fill_value=0)

    pump_flowrates = pd.DataFrame(data=pump_flowrate_raw, index=range(0, timestep * 24, timestep), columns=pump_ids)
    pump_flowrates = pump_flowrates.clip(lower=0)

    calculation = total_energy_and_cost(wn, pump_flowrates, total_heads, pump_id_list, timestep)
    total_energy = calculation[0]
    total_cost = calculation[1]

    critical_node_pressures = []
    for node in range(node_idx):
        pressure = min(result[:, node])
        critical_node_pressures.append(pressure.numpy())

    return total_energy, total_cost, critical_node_pressures


def run_WNTR_model(file, new_pattern_values, electricity_values, continuous=True):
    wn = make_network(file)
    # Minimum and required pressure for water network
    wn.options.hydraulic.viscosity = 1.0
    wn.options.hydraulic.specific_gravity = 1.0
    wn.options.hydraulic.demand_multiplier = 1.0
    wn.options.hydraulic.demand_model = 'DD'
    wn.options.hydraulic.minimum_pressure = 0
    wn.options.hydraulic.required_pressure = 1
    wn.options.hydraulic.pressure_exponent = 0.5
    wn.options.hydraulic.headloss = 'H-W'
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

    # Setting global energy prices since there is no pattern yet.
    # wn.options.energy.global_price = 3.61e-8
    # https://www.researchgate.net/publication/238041923_A_mixed_integer_linear_formulation_for_microgrid_economic_scheduling/figures

    wn.add_pattern('EnergyPrice', electricity_values)
    wn.options.energy.global_pattern = 'EnergyPrice'

    set_pump_efficiency(wn)

    pump_pattern_ids = ['pump_' + item for item in wn.pump_name_list]
    change_pumping_pattern(wn, pump_pattern_ids, new_pattern_values)

    result = simulate_network(wn)

    output = {'wn': wn, 'result': result}

    return output


def run_metamodel(network_name, new_pump_pattern_values):
    # wn = make_network(file)

    datasets_MLP, gn, indices, junctions, tanks, output_nodes, names = prepare_scheduling(
        network_name)

    one_sample = datasets_MLP[0][0]

    if len(new_pump_pattern_values[0]) > 1 and len(new_pump_pattern_values[0]) != 24:
        one_sample = one_sample.repeat(len(new_pump_pattern_values[0]), 1)
    else:
        one_sample = one_sample.unsqueeze(0)

    one_sample[:, indices['pump_schedules']] = torch.tensor(new_pump_pattern_values[0])

    mm = metamodel.MyMetamodel()
    prediction = mm.predict(one_sample)
    pred_formatted = prediction.squeeze().reshape(-1, 1)

    pred = gn.denormalize_multiple(pred_formatted, output_nodes)

    return pred, junctions + tanks, output_nodes, names
