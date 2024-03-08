import wntr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

if __name__ == '__main__':
    # Load your water network model
    inp_file = 'FOS_pump_4.inp'  # Update this with the path to your network file
    wn = wntr.network.WaterNetworkModel(inp_file)

    # Generate incremental node values
    node_values = {
        '1': 0.6928116083145142,
        '2': 0.6423719525337219,
        '3': 0.6529452800750732,
        '4': 0.670391857624054,
        '5': 0.5211731195449829,
        '6': 0.5906476974487305,
        '7': 0.637734591960907,
        '8': 0.6426774263381958,
        '9': 0.6665240526199341,
        '10': 0.6908645629882812,
        '11': 0.688367486000061,
        '12': 0.679815411567688,
        '13': 0.5954548716545105,
        '14': 0.6558042168617249,
        '15': 0.679818868637085,
        '16': 0.6832971572875977,
        '17': 0.5928195118904114,
        '18': 0.6868227124214172,
        '19': 0.683014988899231,
        '20': 0.6693894267082214,
        '21': 0.6477838158607483,
        '22': 0.6784634590148926,
        '23': 0.668782651424408,
        '24': 0.6142408847808838,
        '25': 0.6782899498939514,
        '26': 0.689267635345459,
        '27': 0.6885459423065186,
        '28': 0.6246125102043152,
        '29': 0.6532987356185913,
        '30': 0.6082277297973633,
        '31': 0.6930745244026184,
        '32': 0.6875451803207397,
        '33': 0.6947723627090454,
        '34': 0.6925997734069824,
        '35': 0.6546745300292969,
        '36': 0.687039315700531,
        '39': 0.3465168476104736,
        '58': 0.5114114284515381
    }
    # Normalize the values to get a colormap for nodes, tanks, and pumps
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create a colormap
    cmap = plt.cm.viridis

    # Create a figure and plot the network
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the network's pipes and pumps
    for name, link in wn.links():
        start_node = wn.get_node(link.start_node_name)
        end_node = wn.get_node(link.end_node_name)
        if isinstance(link, wntr.network.Pipe):
            ax.plot([start_node.coordinates[0], end_node.coordinates[0]],
                    [start_node.coordinates[1], end_node.coordinates[1]],
                    color='silver', linewidth=1, zorder=1)
        elif isinstance(link, wntr.network.Pump):
            # For pumps, draw the line and add a special marker in the middle
            mid_point = [(start_node.coordinates[0] + end_node.coordinates[0]) / 2,
                         (start_node.coordinates[1] + end_node.coordinates[1]) / 2]
            ax.plot([start_node.coordinates[0], end_node.coordinates[0]],
                    [start_node.coordinates[1], end_node.coordinates[1]],
                    color='black', linewidth=2, linestyle='--', zorder=1)
            # Pump marker (circle) with color based on its ID value
            pump_value = node_values.get(name, 0)
            pump_color = cmap(norm(pump_value))
            ax.plot(mid_point[0], mid_point[1], '^', color=pump_color, markersize=14, zorder=2, label='Pump')
            ax.text(mid_point[0], mid_point[1], f'{pump_value:.2f}', fontsize=14, ha='right', va='bottom')

    # Plot nodes and annotate them with their values
    for node_name, node_obj in wn.nodes():
        value = node_values.get(node_name, 0)
        color = cmap(norm(value))
        if isinstance(node_obj, wntr.network.Tank):
            # Tank marker (triangle) with color based on its ID value
            ax.plot(node_obj.coordinates[0], node_obj.coordinates[1], 's', color=color, markersize=14, zorder=3, label='Tank')
            ax.text(node_obj.coordinates[0], node_obj.coordinates[1], f'{value:.2f}', fontsize=14, ha='right', va='top')
        elif isinstance(node_obj, wntr.network.Reservoir):
            # Reservoir marker (triangle) with color based on its ID value
            ax.plot(node_obj.coordinates[0], node_obj.coordinates[1], 'o', color=color, markersize=12, zorder=2, label='Reservoir')
        else:  # Other nodes
            ax.plot(node_obj.coordinates[0], node_obj.coordinates[1], 'o', color=color, markersize=12, zorder=2)
            ax.text(node_obj.coordinates[0], node_obj.coordinates[1], f'{value:.2f}', fontsize=14, ha='right', va='top')

    # Add a colorbar to show the mapping from color to value
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower right', fontsize=13)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # Optionally, add titles or labels
    # ax.set_title('Water Network Visualization')
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')

    plt.show()
