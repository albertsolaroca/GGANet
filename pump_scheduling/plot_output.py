import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    ps_output_mm = pd.read_csv('scheduling_mm.csv', index_col=None)
    ps_output_mm = ps_output_mm[ps_output_mm['Valid'] == True]
    ps_output = pd.read_csv('scheduling.csv', index_col=None)

    score_mm = []
    score = []
    avg_score_mm = []
    avg_score = []
    cost = []
    energy = []
    cost_mm = []
    energy_mm = []

    gen_range = range(5, 21, 1)

    for i in gen_range:
        i_gen_mm = ps_output_mm[ps_output_mm['n_generations'] == i]
        total_score_mm = i_gen_mm['Energy (kWh) WNTR'] + i_gen_mm['Cost (€) WNTR']
        total_cost_mm = i_gen_mm['Cost (€) WNTR']
        total_energy_mm = i_gen_mm['Energy (kWh) WNTR']


        i_gen = ps_output[ps_output['n_generations'] == i]
        total_score = i_gen['Energy (kWh) WNTR'] + i_gen['Cost (€) WNTR']
        total_cost = i_gen['Cost (€) WNTR']
        total_energy = i_gen['Energy (kWh) WNTR']

        score_mm.append(total_score_mm.values)
        avg_score_mm.append(np.average(total_score_mm.values))
        score.append(total_score.values)
        avg_score.append(np.average(total_score.values))
        cost.append(np.average(total_cost.values))
        energy.append(np.average(total_energy.values))
        cost_mm.append(np.average(total_cost_mm.values))
        energy_mm.append(np.average(total_energy_mm.values))



    plt.boxplot(score)
    plt.title('Cost + Energy for WNTR')
    plt.show()
    plt.boxplot(score_mm)
    plt.title('Cost + Energy for MM')
    plt.show()

    plt.plot(gen_range, avg_score, label='WNTR')
    plt.plot(gen_range, avg_score_mm, label='MM')
    plt.title('Cost + Energy')
    plt.legend()
    plt.show()
    #
    # plt.plot(gen_range, cost, label='WNTR')
    # plt.plot(gen_range, cost_mm, label='MM')
    # plt.title('Cost')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(gen_range, energy, label='WNTR')
    # plt.plot(gen_range, energy_mm, label='MM')
    # plt.title('Energy')
    # plt.legend()
    # plt.show()

    # plt.plot(ps_output_mm['n_generations'], ps_output_mm['Energy (kWh) WNTR'] + ps_output_mm['Cost (€) WNTR'])
    # plt.plot(ps_output['n_generations'], ps_output['Energy (kWh) WNTR'], label='WNTR')
    # plt.legend()
    # plt.show()

