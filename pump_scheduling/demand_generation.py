"""
The following example demonstrates how to import WNTR, generate a water network
model from an INP file, simulate hydraulics, and plot simulation results on the network.
"""
from matplotlib import pyplot as plt
import pysimdeum
import numpy as np

import wntr
import time

# Demand generation

# Built a house (one-person household)
one_person_house = pysimdeum.built_house(house_type='one_person')
two_person_house = pysimdeum.built_house(house_type='two_person')
family = pysimdeum.built_house(house_type='family')

# Simulate water consumption for house (xarray.DataArray)
consumption_one = one_person_house.simulate(num_patterns=5, duration='1 day')
consumption_two = two_person_house.simulate(num_patterns=10, duration='1 day')
consumption_family = family.simulate(num_patterns=20, duration='1 day')

all_cons = {one_person_house.house_type: consumption_one,
            two_person_house.house_type: consumption_two,
            family.house_type: consumption_family}

tot_avg_cons = {}
tot_rolling_cons = {}
for cons in all_cons:
    averaged_consumption = []
    for i in range(0, 24):
        # average and round to two decimals
        hourly_average_cons = round(float(all_cons[cons][3600 * i:3600 * (i + 1)].sum() / 3600), 2)
        averaged_consumption.append(hourly_average_cons)

    tot_avg_cons[cons] = averaged_consumption



def replace_demand_patterns(file_path, demand_patterns):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    pattern_index = 1
    found_patterns = False
    for i, line in enumerate(lines):
        if line.strip().startswith(';'):
            continue  # Skip comment lines

        if line.strip().startswith('[PATTERNS]'):
            found_patterns = True
        elif found_patterns and line.strip() == '':
            for pattern in demand_patterns:
                pattern_values = ' '.join(map(str, demand_patterns[pattern]))
                lines[i - 1] += f' {pattern}          {"".join(pattern_values)}\n'
            break

    with open(file_path, 'w') as file:
        file.writelines(lines)


# # Usage example
# inp_file_path = './networks/Anytown_test.inp'
#
# replace_demand_patterns(inp_file_path, tot_avg_cons)

plot_colors = ['lightblue','mediumblue','darkblue']
colour_index = 0
for cons in all_cons:
    print(tot_avg_cons[cons])
    # Build statistics from consumption
    tot_cons = all_cons[cons].sum(['enduse', 'user']).sum(['patterns'])
    # rolling = all_cons[cons].rolling(time=3600, center=True).mean()
    # len(tot_cons)
    # Plot total consumption
    # tot_cons.plot()
    # Plot below list in the graph
    # plt.plot(tot_avg_cons[cons])
    rolling_sum = tot_cons.rolling(time=3600, center=True).mean()

    index = rolling_sum.indexes['time']
    # print(rolling_sum.indexes)

    plt.plot(rolling_sum, color=plot_colors[colour_index], label=cons)

    colour_index += 1


# Setting the number of ticks
# plt.xlim(0, 25)
plt.xticks(range(0, 90000, 3600), range(0, 25))
# plt.locator_params(axis='x', nbins=25)
plt.legend()
plt.ylabel("Demand Multiplier")
plt.xlabel("Hour")
plt.show()