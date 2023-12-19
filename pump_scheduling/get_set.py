import wntr


# Converts the input file (usually .inp) to a water network model
def make_network(input_file):
    water_network = wntr.network.WaterNetworkModel(input_file)
    water_network.options.hydraulic.demand_model = 'DD'
    return water_network


# Simulates the networks using the Epanet simulator (instead of WNTRSimulator)
def simulate_network(water_network):
    sim = wntr.sim.EpanetSimulator(water_network)
    results = sim.run_sim()
    return results


# Returns the pressure in the network at a given time
def get_pressure_at_time(results, time):
    pressure = results.node['pressure'].loc[time, :]
    return pressure


# Returns the pressure in the network at a given node
def get_pressure_at_node(results, node):
    pressure = results.node['pressure'].loc[:, node]
    return pressure


# Returns the head in the network at a given time
def get_head_at_time(results, time):
    head = results.node['head'].loc[time, :]
    return head


# Returns the head in the network at a given node
def get_head_at_node(results, node):
    head = results.node['head'].loc[:, node]
    return head


# Returns pump schedule values (as multipliers from 0 to 1) for a given pump
def get_pattern_values(wn, pump_pattern_id):
    pattern = wn.get_pattern(pump_pattern_id)
    pattern_values = pattern.multipliers
    return pattern_values


def get_junction_nodes(wn):
    junction_nodes = wn.junction_name_list
    return junction_nodes

def get_tank_nodes(wn):
    tank_nodes = wn.tank_name_list
    return tank_nodes

def set_pump_efficiency(wn, efficiency=75):
    wn.options.energy.global_efficiency = efficiency
    return


def get_pump_flowrate(results, pump_id):
    pump_flowrate = results.link['flowrate'].loc[:, pump_id]
    return pump_flowrate
