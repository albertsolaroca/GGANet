import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from main_unrolling.training.models import Dummy
from training.test import testing_plain
from utils import load_raw_dataset
from tune_train import prepare_training, default_configuration

def metrics(real, pred, dummy):
    def nse(observed, simulated):
        """
        Calculate Nash-Sutcliffe Efficiency (NSE) for 2D tensors.

        Args:
        observed (torch.Tensor): Tensor containing observed values.
        simulated (torch.Tensor): Tensor containing simulated values.

        Returns:
        NSE (float): Nash-Sutcliffe Efficiency value.
        """
        assert observed.shape == simulated.shape, "Input tensors must have the same shape."

        numerator = torch.sum((observed - simulated) ** 2)
        denominator = torch.sum((observed - torch.mean(observed)) ** 2)

        nse = 1 - (numerator / denominator)
        return nse.item()

    dummy_score = r2_score(real, dummy, multioutput='variance_weighted')
    model_score = r2_score(real, pred, multioutput='variance_weighted')
    print("R2-values \n", "Dummy:", dummy_score, "\n Model", model_score)

    dummy_score = mean_absolute_error(real, dummy)
    model_score = mean_absolute_error(real, pred)
    print("MAE-values \n", "Dummy:", dummy_score, "\n Model", model_score)

    dummy_score = mean_squared_error(real, dummy)
    model_score = mean_squared_error(real, pred)
    print("MSE-values \n", "Dummy:", dummy_score, "\n Model", model_score)

    dummy_score = mean_squared_error(real, dummy, squared=False)
    model_score = mean_squared_error(real, pred, squared=False)
    print("RMSE-values \n", "Dummy:", dummy_score, "\n Model", model_score)

    dummy_score = nse(real, dummy)
    model_score = nse(real, pred)
    print("NSE-values General\n", "Dummy:", dummy_score, "\n Model", model_score)

    dummy_scores = []
    model_scores = []
    for i in range(36):
        dummy_score = nse(real[:, i], dummy[:, i])
        model_score = nse(real[:, i], pred[:, i])
        dummy_scores.append(dummy_score)
        model_scores.append(model_score)
        print("NSE-values for node", i, "\n", "Dummy:", dummy_score, "\n Model", model_score)


if __name__ == "__main__":

    # initialize pytorch device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_folder = '../data_generation/datasets'
    default_config = default_configuration()

    datasets_MLP, gn, indices, junctions, tanks, output_nodes, names = prepare_training(
        default_config.network, default_config.samples)
    # retrieve wntr data
    tra_database, val_database, tst_database = load_raw_dataset(default_config.network, data_folder)

    model_path = 'experiments/unrolling_WDN0120/FOS_pump_sched_flow/UnrollingModel/model.pickle'
    with open(model_path, 'rb') as handle:
        model = torch.load(handle)
        model.eval()

    tst_loader = torch.utils.data.DataLoader(datasets_MLP[2],
                                             batch_size=default_config.batch_size, shuffle=False, pin_memory=True)

    # input = tra_dataset_MLP[0][0].unsqueeze(0).to(device)
    # print(input.shape)
    #
    # start_time = time.time_ns()
    # print(start_time)
    # for batch in tst_loader:
    #     input = batch[0].to(device)
    #     output = model(input)
    #
    # end_time = time.time_ns()
    # print(end_time)
    #
    # print(f"Simulation time: {end_time - start_time}")
    #
    # output = output.detach().cpu().numpy()
    # output = gn.inverse_transform_array(output, 'pressure')
    # print(output)

    pred, real, elapsed_time = testing_plain(model, tst_loader)
    print("TIME TAKEN: ", elapsed_time)
    pred = gn.denormalize_multiple(pred, output_nodes)
    real = gn.denormalize_multiple(real, output_nodes)
    dummy = Dummy(junctions + tanks).evaluate(real)
    # Array below is created to ensure proper indexing of the nodes when displaying
    type_array = (tst_database[0].node_type == 0) | (tst_database[0].node_type == 2)

    for i in [0, 1, 6, 26, 36]:
        plt.plot(real[0:100, i], label="Real")
        plt.plot(pred[0:100, i], label="Predicted")
        plt.plot(dummy[0:100, i], label="Dummy")
        plt.ylabel('Head')
        plt.xlabel('Timestep')

        plt.legend()
        names = {0: 'Next to Reservoir', 1: 'Random Node', 6: 'Next to Tank', 26: 'Random Node', 36: 'Tank', 37: 'Pump'}
        plt.title(names[i])
        # save_response_graphs_in_ML_tracker(real, pred, names[i], i)
        plt.show()
        plt.close()

    plt.plot(real[0:100, 37], label="Real")
    plt.plot(pred[0:100, 37], label="Predicted")
    plt.plot(dummy[0:100, 37], label="Dummy")
    plt.ylabel('LPS')
    plt.xlabel('Timestep')

    plt.legend()
    plt.title(names[37])
    plt.show()
    plt.close()

    metrics(real, pred, dummy)