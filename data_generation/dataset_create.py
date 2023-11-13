import os
import sys
from wntr_utils import import_config, get_wdn_components, create_and_save
import argparse
import traceback
import demand_generation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Argument for continuous (Extended Period Analysis data generation)
    parser.add_argument('-c')
    args = parser.parse_args()

    if args.c == 'Yes' or args.c == 'yes':
        continuous = True
    else:
        continuous = False

    # get configurations
    networks, n_trials = import_config('database_config.yaml')

    path = f'{sys.path[0]}/networks/'
    df_counts = get_wdn_components(networks, path)
    ordered_networks = df_counts.sort_values(by=['nodes']).index.tolist()

    #

    for i, network in enumerate(ordered_networks):
        out_path = os.getcwd() + '\\datasets\\'


        for ix_trials, trials in enumerate(n_trials):
            print(
                f'Working with {network}, network {i + 1} of {len(ordered_networks)} || size: {trials}, {ix_trials + 1} of {len(n_trials)}')
            try:
                if continuous:
                    randomized_demands = demand_generation.pseudogenerate_demand_patterns()
                else:
                    randomized_demands = None

                create_and_save(network, path, n_trials=trials, out_path=out_path,
                                max_fails=100 * trials, continuous=continuous, randomized_demands=randomized_demands)

            except Exception as e:
                print(e)
                traceback.print_exc()
                print("Too many failed simulations")
                break

