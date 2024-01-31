import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    ps_output_mm = pd.read_csv('scheduling_mm.csv', index_col=None)
    ps_output_mm = ps_output_mm[ps_output_mm['Switches'] == 3]

    ps_output = pd.read_csv('scheduling.csv', index_col=None)

    plt.plot(ps_output_mm['n_generations'], ps_output_mm['Energy (kWh) Metamodel'] + ps_output_mm['Cost (€) Metamodel'], label='Metamodel')
    plt.plot(ps_output['n_generations'], ps_output['Energy (kWh)'], label='WNTR')
    plt.legend()
    plt.show()

