import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    ps_output_mm_all = pd.read_csv('scheduling_mm.csv', index_col=None)
    ps_output_mm = ps_output_mm_all[ps_output_mm_all['Valid'] == True]
    ps_output = pd.read_csv('scheduling.csv', index_col=None)

    score_mm = []
    score = []
    avg_energy_mm = []
    avg_energy = []
    avg_cost_mm = []
    avg_cost = []
    cost = []
    energy = []
    cost_mm = []
    energy_mm = []

    gen_range = range(5, 20, 1)

    for i in gen_range:
        i_gen_mm = ps_output_mm[ps_output_mm['n_generations'] == i]
        total_score_mm = i_gen_mm['Energy (kWh) WNTR'] + i_gen_mm['Cost (€) WNTR']
        total_cost_mm = i_gen_mm['Cost (€) WNTR']
        total_energy_mm = i_gen_mm['Energy (kWh) WNTR']

        i_gen = ps_output[ps_output['n_generations'] == i]
        total_score = i_gen['Energy (kWh) WNTR'] + i_gen['Cost (€) WNTR']
        total_cost = i_gen['Cost (€) WNTR']
        total_energy = i_gen['Energy (kWh) WNTR']

        avg_energy_mm.append(np.average(total_energy_mm.values))
        avg_energy.append(np.average(total_energy.values))

        avg_cost_mm.append(np.average(total_cost_mm))
        avg_cost.append(np.average(total_cost))

        cost += list(total_cost.values)
        energy += list(total_energy.values)
        cost_mm += list(total_cost_mm.values)
        energy_mm += list(total_energy_mm.values)

    plt.gcf().set_tight_layout(True)
    plt.plot(gen_range, avg_energy, '-o', label='EPANET', ms=6, linewidth=2.5)
    plt.plot(gen_range, avg_energy_mm, '-o', label='Metamodel', ms=6, linewidth=2.5)
    plt.xlabel('Generations', fontsize=18)
    plt.ylabel('kWh', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=17)
    plt.show()

    plt.gcf().set_tight_layout(True)
    plt.plot(gen_range, avg_cost, '-o', label='EPANET', ms=6, linewidth=2.5)
    plt.plot(gen_range, avg_cost_mm, '-o', label='Metamodel', ms=6, linewidth=2.5)
    plt.xlabel('Generations', fontsize=18)
    plt.ylabel('€', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=17)
    plt.show()

