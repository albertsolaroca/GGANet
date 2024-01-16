"""
The following example demonstrates how to import WNTR, generate a water network
model from an INP file, simulate hydraulics, and plot simulation results on the network.
"""
import wntr
import time

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



    for i in range(0, len(output), 24):
        energy, cost, nodal_pressures = (
            calculate_objective_function_mm(network_file, electricity_price_values, output[i:(24 + i)], node_idx,
                                            names))

        minimum_pressure_required = [14 for i in range(len(nodal_pressures))]

        pressure_surplus = [-nodal_pressures[j] + minimum_pressure_required[j] for j in range(len(nodal_pressures))]

        total_energy.append(energy)
        total_cost.append(cost)
        total_pressure_surplus.append(pressure_surplus)

    return [total_energy, total_cost], total_pressure_surplus


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

    def __init__(self, network_file, n_var=24, n_ieq_constr=36, switch_penalty=0):
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

    def __init__(self, network_file, n_var=24, n_ieq_constr=36, switch_penalty=0):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_ieq_constr=n_ieq_constr,
                         xl=0,
                         xu=1,
                         vtype=bool)

        self.network_file = network_file
        self.switch_penalty = switch_penalty

    def _evaluate(self, x, out, *args, **kwargs):
        # Minimization function
        # evaluation = optimize_pump_schedule_WNTR(self.network_file, [x])
        evaluation = optimize_pump_schedule_metamodel(self.network_file, [x])
        switches = count_switches_2d(x) * 1
        # The objective of the function. Total energy to minimize
        out["F"] = [evaluation[0][0], evaluation[0][1] + self.switch_penalty * count_switches_2d(x)]
        achtien_x = x[17]
        achtien_y = evaluation[0][0][17]
        achtien = evaluation[0][1][17]
        eval = evaluation[1][17]
        # The constraints of the function, as in pressure violations per node
        out["G"] = evaluation[1]


def make_problem(input_file='FOS_pump_sched_flow_single_1'):
    wn = wntr.network.WaterNetworkModel('../data_generation/networks/' + input_file + '.inp')
    time_discrete = int(wn.options.time.duration / wn.options.time.pattern_timestep) + 1 # EPANET bug
    time_discrete = 24
    junctions = len(wn.junction_name_list)
    tanks = len(wn.tank_name_list)

    return SchedulePump(network_file=input_file, n_var=time_discrete, n_ieq_constr=junctions + tanks)


def make_problem_mm(input_file='FOS_pump_sched_flow_single'):
    wn = wntr.network.WaterNetworkModel('../data_generation/networks/' + input_file + '.inp')
    time_discrete = int(wn.options.time.duration / wn.options.time.pattern_timestep)
    junctions = len(wn.junction_name_list)
    tanks = len(wn.tank_name_list)

    return SchedulePumpBatch(network_file=input_file, n_var=time_discrete, n_ieq_constr=junctions + tanks)


if __name__ == "__main__":
    # print(optimize_pump_schedule_WNTR('FOS_pump_sched_flow_single_1', [[1] * 24]))
    # print(optimize_pump_schedule_metamodel('FOS_pump_sched_flow_single', [[1] * 24]))

    # problem = make_problem('FOS_pump_sched_flow')
    # problem = make_problem()
    problem = make_problem_mm()

    algorithm = NSGA2(pop_size=100,
                      sampling=BinaryRandomSampling(),
                      # crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True)

    termination = get_termination("n_gen", 20)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True, )

    pareto_front = res.F
    sorted_indices = pareto_front[:, 0].argsort()
    sorted_solutions = res.X[sorted_indices]
    print(len(sorted_solutions))

    print("Best solution found: %s" % res.X.astype(int))

    evaluation_mm = optimize_pump_schedule_metamodel('FOS_pump_sched_flow_single', [res.X.astype(int)])
    evaluation = optimize_pump_schedule_WNTR('FOS_pump_sched_flow_single_1', res.X.astype(int))

    print("Evaluation of the best solution by mm: %s" % evaluation_mm[0])
    print("Evaluation of the best solution by WNTR: %s" % evaluation[0])
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)
