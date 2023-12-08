# Imports
import numpy as np
import torch
import wntr
import sys
import time

from wntr.network import Pattern
from pump_scheduling.get_set import *
from main_unrolling.tune_train import prepare_training
import pump_scheduling.metamodel as metamodel


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
    energy = wntr.metrics.pump_energy(pump_flowrate, head, wn) # in J
    energy = energy / 3600000 # in kWh

    return energy


# Returns the energy cost of a given pump per timestep
def energy_cost(energy, wn):
    cost = wntr.metrics.pump_cost(energy, wn)
    return cost


# Returns the total energy consumption and cost for a given pump
def total_energy_and_cost(wn, result, pump_id_list, timestep=3600):

    pump_flowrate = result.link['flowrate'].loc[:, wn.pump_name_list]
    # Heads
    head = result.node['head']

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
def calculate_objective_function(wn, result, critical_nodes, pump_id_list):
    calculation = total_energy_and_cost(wn, result, pump_id_list)
    total_energy = calculation[0]
    total_cost = calculation[1]

    critical_node_pressures = []
    for node in critical_nodes:
        pressure = min(get_pressure_at_node(result, node))
        critical_node_pressures.append(pressure)

    return total_energy, total_cost, critical_node_pressures


def run_WNTR_model(file, new_pattern_values, electricity_values):
    wn = make_network(file)


    # Minimum and required pressure for water network
    wn.options.hydraulic.required_pressure = 14
    wn.options.hydraulic.minimum_pressure = 0

    # Setting global energy prices since there is no pattern yet.
    # wn.options.energy.global_price = 3.61e-8
    wn.options.energy.global_pattern = electricity_values
    #https://www.researchgate.net/publication/238041923_A_mixed_integer_linear_formulation_for_microgrid_economic_scheduling/figures

    wn.add_pattern('EnergyPrice', electricity_values)
    wn.options.energy.global_pattern = 'EnergyPrice'

    set_pump_efficiency(wn)

    pump_pattern_ids = ['pump_' + item for item in wn.pump_name_list]
    change_pumping_pattern(wn, pump_pattern_ids, new_pattern_values)

    result = simulate_network(wn)

    critical_nodes = get_junction_nodes(wn)

    output = {'wn': wn, 'result': result, 'critical_nodes': critical_nodes}

    return output

def run_metamodel(network_name, new_pattern_values, electricity_values):

    # wn = make_network(file)

    tra_dataset_MLP, val_dataset_MLP, tst_dataset_MLP, gn, indices, junctions, output_nodes = prepare_training(
        network_name, 1)

    one_sample = tra_dataset_MLP[0][0].unsqueeze(0)

    mm = metamodel.MyMetamodel()
    prediction = mm.predict(one_sample)
    output_norm = prediction.squeeze()
    # output_norm = prediction.reshape(-1, len(prediction[0][0]))

    output = gn.inverse_transform_array(output_norm, 'pressure', reshape=False)

    # result = simulate_network(wn)
    #
    # critical_nodes = get_junction_nodes(wn)
    #
    # output = {'wn': wn, 'result': result, 'critical_nodes': critical_nodes}

    return None
