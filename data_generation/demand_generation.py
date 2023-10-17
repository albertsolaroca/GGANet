"""
The following example demonstrates how to import WNTR, generate a water network
model from an INP file, simulate hydraulics, and plot simulation results on the network.
"""
import threading

from matplotlib import pyplot as plt
import pysimdeum

def simulate_thread(house_type, num_patterns, duration, output):
    house = pysimdeum.built_house(house_type=house_type)
    consumption = house.simulate(num_patterns=num_patterns, duration=duration)
    output[house_type] = consumption

def generate_demand_patterns():

    # Demand generation

    # Build houses
    # Simulate water consumption for house (xarray.DataArray)
    all_cons = {}
    # Create threads for each simulation
    thread1 = threading.Thread(target=simulate_thread, args=('one_person', 25, '1 day', all_cons))
    thread2 = threading.Thread(target=simulate_thread, args=('two_person', 50, '1 day', all_cons))
    thread3 = threading.Thread(target=simulate_thread, args=('family', 80, '1 day', all_cons))

    # Start the threads
    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for all threads to finish
    thread1.join()
    thread2.join()
    thread3.join()

    tot_avg_cons = {}
    tot_rolling_cons = {}
    for cons in all_cons:
        averaged_consumption = []
        for i in range(0, 24):
            # average and round to two decimals
            hourly_average_cons = round(float(all_cons[cons][3600 * i:3600 * (i + 1)].sum() / 3600), 2)
            averaged_consumption.append(hourly_average_cons)

        tot_avg_cons[cons] = averaged_consumption

    # Write tot_avg_cons to yaml file without yaml packages
    with open('demand_patterns.yaml', 'w') as file:
        file.write('demand_patterns:\n')
        for cons in tot_avg_cons:
            file.write(f'  {cons}: {tot_avg_cons[cons]}\n')




    return tot_avg_cons

    # # Plot the consumption patterns
    # plot_colors = ['lightblue', 'mediumblue', 'darkblue']
    # colour_index = 0
    # for cons in all_cons:
    #     print(tot_avg_cons[cons])
    #     # Build statistics from consumption
    #     tot_cons = all_cons[cons].sum(['enduse', 'user']).sum(['patterns'])
    #     rolling_sum = tot_cons.rolling(time=3600, center=True).mean()
    #
    #     index = rolling_sum.indexes['time']
    #
    #     plt.plot(rolling_sum, color=plot_colors[colour_index], label=cons)
    #
    #     colour_index += 1
    #
    # # Setting the number of ticks
    # plt.xticks(range(0, 90000, 3600), range(0, 25))
    # plt.legend()
    # plt.ylabel("Demand Multiplier")
    # plt.xlabel("Hour")
    # plt.show()

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



