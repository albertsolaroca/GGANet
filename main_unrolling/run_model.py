import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from training.models import Dummy
from training.test import testing_plain
from utils.load import load_raw_dataset
from tune_train import prepare_training, default_configuration


def metrics(real, pred):
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

    model_scores = []
    for i in range(len(real[0])):
        model_score = nse(real[:, i], pred[:, i])
        model_scores.append(model_score)

    plt.plot(model_scores, '-o', label="Model", c='darkorange', ms=2)
    plt.ylabel('R2', fontsize=16)
    plt.xlabel('Node', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gcf().set_tight_layout(True)
    plt.savefig('C:/Users/nmert/OneDrive/Pictures/Thesis/ky2_c/script/' + "ModelR2")
    plt.close()

if __name__ == "__main__":

    # initialize pytorch device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_folder = '../data_generation/datasets'
    default_config = default_configuration()

    datasets_MLP, gn, indices, junctions, tanks, output_nodes, names = prepare_training(
        default_config.network, default_config.samples)
    # retrieve wntr data
    tra_database, val_database, tst_database = load_raw_dataset(default_config.network, data_folder)

    model_path = 'experiments/unrolling_WDN0396/ky2_c/UnrollingModel/model.pickle'

    with open(model_path, 'rb') as handle:
        model = torch.load(handle)
        model.eval()

    tst_loader = torch.utils.data.DataLoader(datasets_MLP[2],
                                             batch_size=default_config.batch_size, shuffle=False, pin_memory=True)

    pred, real, elapsed_time = testing_plain(model, tst_loader)
    print("TIME TAKEN: ", elapsed_time)
    pred = gn.denormalize_multiple(pred, output_nodes)
    real = gn.denormalize_multiple(real, output_nodes)
    dummy = Dummy(junctions + tanks).evaluate(real)
    # Array below is created to ensure proper indexing of the nodes when displaying
    type_array = (tst_database[0].node_type == 0) | (tst_database[0].node_type == 2)

    mpl.rcParams["font.size"] = 16
    for i in [0, 54, 116, 235, 387, 110, 482]:
        plt.plot(real[0:100, i], label="Real")
        plt.plot(pred[0:100, i], label="Predicted")

        plt.ylabel('Head')
        plt.xlabel('Timestep')

        plt.legend()
        names = {0: 'Random Node', 54: 'Random Node', 116: 'Random Node', 235: 'Random Node', 387: 'Random Node', 110: 'Random Node', 482: 'Tank'}
        plt.gcf().set_tight_layout(True)
        plt.savefig('C:/Users/nmert/OneDrive/Pictures/Thesis/ky2_c/script/' + names[i] + ' - ' + str(i + 1))
        plt.close()


    plt.plot(real[0:100, len(real[0]) - 2], label="Real")
    plt.plot(pred[0:100, len(real[0]) - 2], label="Predicted")
    plt.ylabel('Head')
    plt.xlabel('Timestep')
    plt.gcf().set_tight_layout(True)
    plt.legend()
    plt.savefig('C:/Users/nmert/OneDrive/Pictures/Thesis/ky2_c/script/' + "Next to pump - " + str(len(real[0]) - 2))
    plt.close()


    plt.plot(real[0:100, len(real[0]) - 1], label="Real")
    plt.plot(pred[0:100, len(real[0]) - 1], label="Predicted")
    # plt.plot(dummy[0:100, len(real[0]) - 1], label="Dummy")
    plt.ylabel('LPS')
    plt.xlabel('Timestep')
    plt.gcf().set_tight_layout(True)
    plt.legend()
    plt.savefig('C:/Users/nmert/OneDrive/Pictures/Thesis/ky2_c/script/' + "Pump - " + str(len(real[0]) - 1))
    plt.close()

    metrics(real, pred)


def extra():
    def metrics(real, pred):
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

        model_scores = []
        for i in range(36):
            model_score = nse(real[:, i], pred[:, i])
            model_scores.append(model_score)

    if __name__ == "__main__":

        # initialize pytorch device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_folder = '../data_generation/datasets'
        default_config = default_configuration()

        datasets_MLP, gn, indices, junctions, tanks, output_nodes, names = prepare_training(
            default_config.network, 20)
        # retrieve wntr data
        tra_database, val_database, tst_database = load_raw_dataset(default_config.network, data_folder)

        model_path = 'experiments/unrolling_WDN0373/EPANET Net 3/UnrollingModel/model.pickle'

        with open(model_path, 'rb') as handle:
            model = torch.load(handle)
            model.eval()

        tst_loader = torch.utils.data.DataLoader(datasets_MLP[2],
                                                 batch_size=default_config.batch_size, shuffle=False, pin_memory=True)

        pred, real, elapsed_time = testing_plain(model, tst_loader)
        print("TIME TAKEN: ", elapsed_time)
        pred = gn.denormalize_multiple(pred, output_nodes)
        real = gn.denormalize_multiple(real, output_nodes)
        dummy = Dummy(junctions + tanks).evaluate(real)
        # Array below is created to ensure proper indexing of the nodes when displaying
        type_array = (tst_database[0].node_type == 0) | (tst_database[0].node_type == 2)
        mpl.rcParams["font.size"] = 16
        for i in [0, 6, 26, 37, 106]:
            plt.plot(real[0:100, i], label="Real")
            plt.plot(pred[0:100, i], label="Predicted")

            plt.ylabel('Head')
            plt.xlabel('Hour')

            plt.legend()
            names = {0: 'Tank', 1: 'Tank', 2: 'Tank', 6: 'Random Node', 26: 'Random Node', 36: 'Random Node',
                     37: 'Random Node', 106: 'Random Node'}
            plt.gcf().set_tight_layout(True)
            plt.savefig('C:/Users/nmert/OneDrive/Pictures/Thesis/EpanetNet3/script/' + names[i] + ' - ' + str(i + 1))
            plt.close()

        plt.plot(real[0:100, len(real[0]) - 2], label="Real")
        plt.plot(pred[0:100, len(real[0]) - 2], label="Predicted")
        plt.ylabel('LPS')
        plt.xlabel('Timestep')
        plt.gcf().set_tight_layout(True)
        plt.legend()
        plt.savefig('C:/Users/nmert/OneDrive/Pictures/Thesis/EpanetNet3/script/' + "Pump - " + str(len(real[0]) - 2))
        plt.close()

        metrics(real, pred)