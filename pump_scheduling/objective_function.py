# Imports
import numpy as np
import pandas as pd
import torch
import time

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


def pump_energy_mm(flowrate, head):

    power = pump_power_mm(flowrate, head)  # Watts = J/s
    energy = power * 3600  # J = Ws

    return energy


def pump_power_mm(flowrate, head, start_node=36, end_node=0, efficiency=75):

    start_head = head[:, start_node]
    end_head = head[:, end_node]
    headloss = end_head - start_head

    power = 1000.0 * 9.81 * headloss * flowrate / efficiency  # Watts = J/s

    return power

def energy_consumption_mm(pump_flowrate, head):
    # Energy consumption in kWh
    energy = pump_energy_mm(pump_flowrate, head)  # in J
    energy = energy / 3600000  # in kWh

    return energy

# Returns the energy cost of a given pump per timestep
def energy_cost(energy, wn):
    cost = wntr.metrics.pump_cost(energy, wn)
    return cost

def energy_cost_mm(energy, electricity_values):
    cost = energy * electricity_values
    return np.sum(cost)


# Returns the total energy consumption and cost for a given pump
def total_energy_and_cost(wn, pump_flowrate, head, pump_id_list, timestep=3600):
    energy = energy_consumption(wn, pump_flowrate, head)
    if len(energy) != 24:
        print('wat')
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

        total_energy_per_pump.append(np.sum(pump_energy_series))
        total_cost_per_pump.append(np.sum(pump_cost_series))

    total_energy = np.sum(total_energy_per_pump, axis=0)
    total_cost = np.sum(total_cost_per_pump, axis=0)

    return total_energy, total_cost

def total_energy_and_cost_mm(pump_flowrate, head, electricity_values):
    energy = energy_consumption_mm(pump_flowrate, head)
    if len(energy) != 24:
        print('wat')
    cost = energy_cost_mm(energy, electricity_values)

    total_energy = np.sum(energy, axis=0)
    total_cost = np.sum(cost, axis=0)

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
from numba import jit
# @jit(nopython=True)
def calculate_objective_function_mm(result, node_idx, electricity_values):

    # Vectorized operation to find the minimum pressure for each node
    min_pressures = result[:, :node_idx].min(axis=0)  # Min along rows for each column (node)
    critical_node_pressures = -min_pressures[:node_idx] + 15  # Assuming node_idx is the count of nodes
    criticality = determine_positive(critical_node_pressures)

    if criticality == 1:
        return 0, 0, 1

    pump_flowrate = np.squeeze(result[:, node_idx:])
    total_heads = result[:, :node_idx]

    calculation = total_energy_and_cost_mm(pump_flowrate, total_heads, electricity_values)
    total_energy = calculation[0]
    total_cost = calculation[1]

    return total_energy, total_cost, criticality

def determine_positive(lst):
    # Convert the list to a NumPy array for efficient processing
    # This conversion is efficient; NumPy is optimized for such operations
    return 1 if np.any(lst > 0) else -1

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

    set_pump_efficiency(wn, 75)

    pump_pattern_ids = ['pump_' + item for item in wn.pump_name_list]
    change_pumping_pattern(wn, pump_pattern_ids, new_pattern_values)

    time_start = time.time()
    result = simulate_network(wn)
    time_end = time.time()
    # print("Time for simulation of WNTR:", time_end - time_start)

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
