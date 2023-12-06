"""
The following example demonstrates how to import WNTR, generate a water network
model from an INP file, simulate hydraulics, and plot simulation results on the network.
"""
from matplotlib import pyplot as plt
import pysimdeum
import wntr
import time

# Create a water network model
inp_file = '../networks/FOS_pump_0.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

# Graph the network
wntr.graphics.plot_network(wn, title=wn.name)
print(wn.pattern_name_list)
# Simulate hydraulics
start1 = time.time()
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
end1 = time.time()
print('Time to simulate: ', end1 - start1)

start2 = time.time()
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()
end2 = time.time()
print('Time to simulate: ', end2 - start2)


# Plot results on the network
pressure_at_5hr = results.node['pressure'].loc[6 * 3600, :]
wntr.graphics.plot_network(wn, node_attribute=pressure_at_5hr, node_size=30,
                           title='Pressure at 6 hours')

# Theres demand, head, pressure and quality in results

# Get pressure per node
pressure_at_5hr = results.node['pressure'].loc[6 * 3600, :]
print('Pressure at 6 hr per node: \n', pressure_at_5hr)

# Get head per node
head_at_5hr = results.node['head'].loc[6 * 3600, :]
print('Head at 6 hr per node: \n', head_at_5hr)

plt.show()
