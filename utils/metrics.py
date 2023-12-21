from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch

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


def calculate_metrics(real, dummy, pred):

    r2_dummy_score = r2_score(real, dummy, multioutput='variance_weighted')
    r2_model_score = r2_score(real, pred, multioutput='variance_weighted')

    mae_dummy_score = mean_absolute_error(real, dummy)
    mae_model_score = mean_absolute_error(real, pred)

    mse_dummy_score = mean_squared_error(real, dummy)
    mse_model_score = mean_squared_error(real, pred)

    rmse_dummy_score = mean_squared_error(real, dummy, squared=False)
    rmse_model_score = mean_squared_error(real, pred, squared=False)

    nse_dummy_score = nse(real, dummy)
    nse_model_score = nse(real, pred)

    nse_dummy_scores = []
    nse_model_scores = []
    for i in range(len(real[0])):
        dummy_score = nse(real[:, i], dummy[:, i])
        model_score = nse(real[:, i], pred[:, i])
        nse_dummy_scores.append(dummy_score)
        nse_model_scores.append(model_score)

    summed_scores = {'r2_dummy_score': r2_dummy_score, 'r2_model_score': r2_model_score,
                     'mae_dummy_score': mae_dummy_score, 'mae_model_score': mae_model_score,
                     'mse_dummy_score': mse_dummy_score, 'mse_model_score': mse_model_score,
                     'rmse_dummy_score': rmse_dummy_score, 'rmse_model_score': rmse_model_score,
                     'nse_dummy_score': nse_dummy_score, 'nse_model_score': nse_model_score}

    return summed_scores, nse_dummy_scores, nse_model_scores