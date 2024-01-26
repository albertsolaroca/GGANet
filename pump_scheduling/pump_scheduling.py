"""
The following example demonstrates how to import WNTR, generate a water network
model from an INP file, simulate hydraulics, and plot simulation results on the network.
"""
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import wntr
import os
import time
import itertools

from pymoo.algorithms.moo.nsga3 import NSGA3

from objective_function import calculate_objective_function, calculate_objective_function_mm, run_WNTR_model, \
    run_metamodel
import numpy as np

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.operators.sampling.rnd import BinaryRandomSampling

from get_set import make_network


def optimize_pump_schedule_WNTR(network_file, new_pump_pattern_values):
    network_file = '../data_generation/networks/' + network_file + '.inp'
    # Pre-set electricity pattern values to calculate cost
    electricity_pattern_values = [0.065, 0.06, 0.045, 0.047, 0.049, 0.07, 0.085, 0.09, 0.14, 0.19, 0.1, 0.11, 0.125,
                                  0.095, 0.085, 0.08, 0.087, 0.087, 0.09, 0.09, 0.083, 0.18, 0.06, 0.04]
    # in $/kWh
    # https://www.researchgate.net/publication/238041923_A_mixed_integer_linear_formulation_for_microgrid_economic_scheduling/figures

    # Run the model with the pump pattern values specified
    output = run_WNTR_model(network_file, new_pump_pattern_values, electricity_pattern_values)

    # Calculate the objective function, which is the total energy, cost and minimum pressure per node during the run
    results = calculate_objective_function(output['wn'], output['result'])

    total_energy = results[0]
    total_cost = results[1]
    nodal_pressures = results[2]
    minimum_pressure_required = [14 for i in range(len(nodal_pressures))]

    pressure_surplus = [-nodal_pressures[j] + minimum_pressure_required[j] for j in range(len(nodal_pressures))]

    return [total_energy, total_cost], pressure_surplus


def optimize_pump_schedule_metamodel(network_file, new_pump_pattern_values):
    # Pre-set electricity pattern values to calculate cost
    electricity_price_values = [0.065, 0.06, 0.045, 0.047, 0.049, 0.07, 0.085, 0.09, 0.14, 0.19, 0.1, 0.11, 0.125,
                                0.095, 0.085, 0.08, 0.087, 0.087, 0.09, 0.09, 0.083, 0.18, 0.06, 0.04]
    # in $/kWh
    # https://www.researchgate.net/publication/238041923_A_mixed_integer_linear_formulation_for_microgrid_economic_scheduling/figures

    # Run the model with the pump pattern values specified
    output, node_idx, pumps_idx, names = run_metamodel(network_file, new_pump_pattern_values)
    # Calculate the objective function, which is the total energy, cost and minimum pressure per node during the run
    total_energy = []
    total_cost = []
    total_pressure_surplus = []

    wn = make_network('../data_generation/networks/' + network_file + '.inp')
    # get wn patterns
    pats = wn.patterns

    wn.add_pattern('EnergyPrice', electricity_price_values)
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

    pump_flowrate_raw = output[:, node_idx:] / 1000 # L/s to m3/s
    pressures_raw = output[:, :node_idx]

    # Heads
    pressures = pd.DataFrame(data=pressures_raw, index=range(0, len(output)), columns=jt_ids)
    heads = pd.DataFrame(data=names['node_heads'], index=range(0, len(output)))

    total_heads = heads.add(pressures, fill_value=0)

    pump_flowrates = pd.DataFrame(data=pump_flowrate_raw, columns=pump_ids)
    pump_flowrates = pump_flowrates.clip(lower=0)

    # Loop through all outputted examples
    for i in range(0, len(output), 24):
        energy, cost, pressure_surplus = (
            calculate_objective_function_mm(wn, output[i:(24 + i)], node_idx, pump_id_list, pump_flowrates[i:(24 + i)], total_heads[i:(24 + i)]))


        total_energy.append(energy)
        total_cost.append(cost)
        total_pressure_surplus.append(pressure_surplus)

    return [total_energy, total_cost], total_pressure_surplus

def optimize_pump_schedule_metamodel_parallel(network_file, new_pump_pattern_values):
    # Pre-set electricity pattern values to calculate cost
    electricity_price_values = [0.065, 0.06, 0.045, 0.047, 0.049, 0.07, 0.085, 0.09, 0.14, 0.19, 0.1, 0.11, 0.125,
                                0.095, 0.085, 0.08, 0.087, 0.087, 0.09, 0.09, 0.083, 0.18, 0.06, 0.04]
    # in $/kWh
    # https://www.researchgate.net/publication/238041923_A_mixed_integer_linear_formulation_for_microgrid_economic_scheduling/figures

    # Run the model with the pump pattern values specified
    output, node_idx, pumps_idx, names = run_metamodel(network_file, new_pump_pattern_values)
    # Calculate the objective function, which is the total energy, cost and minimum pressure per node during the run
    total_energy = []
    total_cost = []
    total_pressure_surplus = []


    wn = make_network('../data_generation/networks/' + network_file + '.inp')
    # get wn patterns
    pats = wn.patterns

    wn.add_pattern('EnergyPrice', electricity_price_values)
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

    pump_flowrate_raw = output[:, node_idx:] / 1000 # L/s to m3/s
    pressures_raw = output[:, :node_idx]

    # Heads
    pressures = pd.DataFrame(data=pressures_raw, index=range(0, len(output)), columns=jt_ids)
    heads = pd.DataFrame(data=names['node_heads'], index=range(0, len(output)))

    total_heads = heads.add(pressures, fill_value=0)

    pump_flowrates = pd.DataFrame(data=pump_flowrate_raw, columns=pump_ids)
    pump_flowrates = pump_flowrates.clip(lower=0)

    # Determine the number of splits based on CPU count
    num_cores = os.cpu_count()
    chunk_size = len(output) // num_cores

    # Splitting DataFrame into chunks
    chunks_pump = [pump_flowrates[i:i + chunk_size] for i in range(0, pump_flowrates.shape[0], chunk_size)]
    # Splitting DataFrame into chunks
    chunks_head = [total_heads[i:i + chunk_size] for i in range(0, total_heads.shape[0], chunk_size)]
    # Splitting output into chunks
    chunks_output = [output[i:i + chunk_size] for i in range(0, output.shape[0], chunk_size)]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks_output, chunks_pump, chunks_head, [node_idx] * chunk_size, [pump_id_list] * chunk_size, [wn] * chunk_size))

    # Assuming results is a list of tuples, each containing three lists
    # results = [(list1a, list1b, list1c), (list2a, list2b, list2c), ...]

    # Extract and concatenate lists from each tuple
    total_energy = list(itertools.chain(*[result[0] for result in results]))
    total_cost = list(itertools.chain(*[result[1] for result in results]))
    total_pressure_surplus = list(itertools.chain(*[result[2] for result in results]))


    return [total_energy, total_cost], total_pressure_surplus

def process_chunk(output, pump_flowrates, total_heads, node_idx, pump_id_list, wn):

    chunk_energy = []
    chunk_cost = []
    chunk_pressure_surplus = []

    for i in range(0, len(output), 24):
        energy, cost, pressure_surplus = (
            calculate_objective_function_mm(wn, output[i:(24 + i)], node_idx, pump_id_list, pump_flowrates[i:(24 + i)], total_heads[i:(24 + i)]))
        chunk_energy.append(energy)
        chunk_cost.append(cost)
        chunk_pressure_surplus.append(pressure_surplus)

    return chunk_energy, chunk_cost, chunk_pressure_surplus

def check_greater_than_zero(lst):
    return 1 if any(x > 0 for x in lst) else -1

def count_switches(numbers):
    numbers = np.array(numbers)
    # Counting switches from 0 to 1
    switches = np.sum((numbers[:-1] == 0) & (numbers[1:] == 1))
    return switches


def count_switches_2d(matrix):
    matrix = np.array(matrix)
    # Counting switches from 0 to 1 for each row
    switches_per_row = np.sum((matrix[:, :-1] == 0) & (matrix[:, 1:] == 1), axis=1)
    return switches_per_row


class SchedulePump(ElementwiseProblem):

    def __init__(self, network_file, n_var=24, n_ieq_constr=37, switch_penalty=0):
        super().__init__(n_var=n_var,
                         n_obj=3,
                         n_ieq_constr=n_ieq_constr,
                         xl=0,
                         xu=1,
                         vtype=bool)

        self.network_file = network_file
        self.switch_penalty = switch_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        # Minimization function
        evaluation = optimize_pump_schedule_WNTR(self.network_file, [x])
        # evaluation = optimize_pump_schedule_metamodel(self.network_file, [x])
        # The objective of the function. Total energy and total cost to minimize
        # The cost is modified with a switch penalty term
        out["F"] = [evaluation[0][0], evaluation[0][1], self.switch_penalty * count_switches(x)]

        # The constraints of the function, as in pressure violations per node
        out["G"] = evaluation[1]


class SchedulePumpBatch(Problem):

    def __init__(self, network_file, n_var=24, n_ieq_constr=37, switch_penalty=0):
        super().__init__(n_var=n_var,
                         n_obj=3,
                         n_ieq_constr=n_ieq_constr,
                         xl=0,
                         xu=1,
                         vtype=bool)

        self.network_file = network_file
        self.switch_penalty = switch_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        # Minimization function
        evaluation = optimize_pump_schedule_metamodel(self.network_file, [x])
        # The objective of the function. Total energy to minimize
        out["F"] = [evaluation[0][0], evaluation[0][1], self.switch_penalty * count_switches_2d(x)]

        # eval = determine_positive(evaluation[1])
        # The constraints of the function, as in pressure violations per node
        out["G"] = evaluation[1]

class SchedulePumpParallel(Problem):

    def __init__(self, network_file, n_var=24, n_ieq_constr=1, switch_penalty=0):
        super().__init__(n_var=n_var,
                         n_obj=3,
                         n_ieq_constr=n_ieq_constr,
                         xl=0,
                         xu=1,
                         vtype=bool)

        self.network_file = network_file
        self.switch_penalty = switch_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        # Minimization function
        evaluation = optimize_pump_schedule_metamodel_parallel(self.network_file, [x])
        # The objective of the function. Total energy to minimize
        out["F"] = [evaluation[0][0], evaluation[0][1], self.switch_penalty * count_switches_2d(x)]

        # The constraints of the function, as in pressure violations per node
        out["G"] = evaluation[1]

def make_problem(input_file='FOS_pump_sched_flow_single_1', switch_penalty=0):
    wn = wntr.network.WaterNetworkModel('../data_generation/networks/' + input_file + '.inp')
    # time_discrete = int(wn.options.time.duration / wn.options.time.pattern_timestep)  + 1 # EPANET bug
    time_discrete = 24
    junctions = len(wn.junction_name_list)
    tanks = len(wn.tank_name_list)

    return SchedulePump(network_file=input_file, n_var=time_discrete, n_ieq_constr=junctions + tanks, switch_penalty=switch_penalty)


def make_problem_mm(input_file='FOS_pump_sched_flow_single', switch_penalty=0):
    wn = wntr.network.WaterNetworkModel('../data_generation/networks/' + input_file + '.inp')
    time_discrete = int(wn.options.time.duration / wn.options.time.pattern_timestep)
    junctions = len(wn.junction_name_list)
    tanks = len(wn.tank_name_list)

    return SchedulePumpBatch(network_file=input_file, n_var=time_discrete, n_ieq_constr=1, switch_penalty=switch_penalty)

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

if __name__ == "__main__":

    # print(optimize_pump_schedule_WNTR('FOS_pump_sched_flow_single_1', [[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    # print(optimize_pump_schedule_metamodel('FOS_pump_sched_flow_single', [[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

    # print(optimize_pump_schedule_WNTR('FOS_pump_sched_flow_single_1', [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]]))
    # print(optimize_pump_schedule_metamodel('FOS_pump_sched_flow_single', [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]]))

    # problem = make_problem()
    # problem = make_problem(input_file='FOS_pump_2_0', switch_penalty=1)
    problem = make_problem_mm(input_file='FOS_pump_2', switch_penalty=1)
    # problem = make_problem_mm(switch_penalty=1)

    algorithm = NSGA2(pop_size=100,
                      sampling=BinaryRandomSampling(),
                      # crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True)

    termination = get_termination("n_gen", 10)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True, )

    pareto_front = res.F
    sorted_indices = pareto_front[:, 0].argsort()
    sorted_solutions = res.X[sorted_indices]
    print('Found', len(sorted_solutions), 'solutions')

    print("Best solution found: %s" % res.X.astype(int))

    if res.X.astype(int).shape[0] > 0:
        for i in range(len(res.X.astype(int))):
            resuts = res.X.astype(int)[i]
            evaluation_mm = optimize_pump_schedule_metamodel('FOS_pump_2', [res.X.astype(int)[i]])
            evaluation = optimize_pump_schedule_WNTR('FOS_pump_2_0', [res.X.astype(int)[i]])
            pressure_issues_mm = [i for i in evaluation_mm[1] if i >= 0]
            print(f"Evaluation of solution {i} by mm: %s" % evaluation_mm[0], pressure_issues_mm)
            pressure_issues = [i for i in evaluation[1] if i >= 0]
            print(f"Evaluation of solution {i} by WNTR: %s" % evaluation[0], pressure_issues)
            print("Function value: %s" % res.F[i])
            print("Constraint violation: %s" % res.CV[i])

    print("Function value: %s" % res.F)

    # t = res.F[:, 0]
    # x = res.F[:, 1]
    # y = res.F[:, 2]
    # # Example data points
    # X_data = np.array([1, 2, 3, 4])
    # Y_data = np.array([2, 3, 4, 5])
    # Z_data = np.array([3, 4, 5, 6])
    #
    # # Create a meshgrid
    # X_range = np.linspace(min(X_data) - 1, max(X_data) + 1, 30)
    # Y_range = np.linspace(min(Y_data) - 1, max(Y_data) + 1, 30)
    # X, Y = np.meshgrid(X_range, Y_range)
    # R = np.sqrt(X ** 2 + Y ** 2)
    #
    # # Define a function for Z
    # Z = np.sqrt(X ** 2 + Y ** 2)
    #
    # # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the surface
    # ax.plot_surface(X, Y, R, alpha=0.7)
    #
    # # Plot data points
    # ax.scatter(X_data, Y_data, Z_data, color='r')
    #
    #
    # # Set labels with increased labelpad
    # ax.set_xlabel('X axis', labelpad=30)
    # ax.set_ylabel('Y axis', labelpad=30)
    # ax.set_zlabel('Z axis', labelpad=30)
    #
    # # Change perspective
    # elevation_angle = 5
    # azimuth_angle = 15
    # ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    #
    # # Legend
    # ax.legend()
    #
    # plt.show()