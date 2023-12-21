# Libraries
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


def parameters_variations(model, load_path):
    '''
    Determine how the parameters change after training
    if the variation is small, it means that there are too many parameters
    ------
    model: 
    '''
    model.eval()
    
    init_weights = []
    for p in model.parameters():
        init_weights.append(p.clone().detach().reshape(-1))

    model.load_state_dict(torch.load(load_path))

    final_weights = []
    for p in model.parameters():
        final_weights.append(p.clone().detach().reshape(-1))

    diff_weights = (torch.cat(final_weights)-torch.cat(init_weights)).numpy()

    plt.hist(diff_weights, bins=100)
    plt.title('Parameter variation after training')
    plt.xlabel('Variations')
    plt.ylabel('Occurences')
    xlim = diff_weights.std()*5
    plt.xlim(-xlim,xlim);
    
    total_parameters = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ",total_parameters)