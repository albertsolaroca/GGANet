import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    ps_output_mm = pd.read_csv('scheduling_mm.csv', index_col=None)
    ps_output_mm = ps_output_mm[ps_output_mm['Valid'] == True]
    ps_output = pd.read_csv('scheduling.csv', index_col=None)

    boxplot_mm = []
    boxplot = []

    for i in range(5, 20, 1):
        i_gen_mm = ps_output_mm[ps_output_mm['n_generations'] == i]
        total_score_mm = i_gen_mm['Energy (kWh) WNTR'] + i_gen_mm['Cost (€) WNTR']

        i_gen = ps_output[ps_output['n_generations'] == i]
        total_score = i_gen['Energy (kWh) WNTR'] + i_gen['Cost (€) WNTR']

        boxplot_mm.append(total_score_mm.values)
        boxplot.append(total_score.values)



    plt.boxplot(boxplot)
    plt.show()
    plt.boxplot(boxplot_mm)
    plt.show()


    # plt.plot(ps_output_mm['n_generations'], ps_output_mm['Energy (kWh) WNTR'] + ps_output_mm['Cost (€) WNTR'])
    # plt.plot(ps_output['n_generations'], ps_output['Energy (kWh) WNTR'], label='WNTR')
    # plt.legend()
    # plt.show()

