import sys
from wntr_utils import *
import argparse
import traceback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Argument for continuous (Extended Period Analysis data generation
    parser.add_argument('-c')
    args = parser.parse_args()

    if args.c == 'Yes' or args.c == 'yes':
        continuous = True
    else:
        continuous = False

    # get configurations
    networks, n_trials, d_attr, d_netw = import_config('database_config.yaml')

    path = f'{sys.path[0]}/networks/'
    df_counts = get_wdn_components(networks, path)
    ordered_networks = df_counts.sort_values(by=['nodes']).index.tolist()

    for i, network in enumerate(ordered_networks):
        out_path = os.getcwd() + '\\datasets\\'
        for ix_trials, trials in enumerate(n_trials):
            print(
                f'Working with {network}, network {i + 1} of {len(ordered_networks)} || size: {trials}, {ix_trials + 1} of {len(n_trials)}')
            try:
                create_and_save(network, path, n_trials=trials, d_attr=d_attr, d_netw=d_netw, out_path=out_path,
                                max_fails=10 * trials, show=True, continuous=continuous)
            except Exception as e:
                print(e)
                traceback.print_exc()
                print("Too many failed simulations")
                break
