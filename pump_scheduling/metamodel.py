import torch

class MyMetamodel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_path = '../main_unrolling/experiments/unrolling_WDN0350/FOS_pump_2/UnrollingModel/model.pickle'
        # If the model above is activated then the following line should be commented out
        # and the num_blocks in the training/models file should be increased by 1 and the network loaded should be changed to FOS_pump_2
        # model_path = '../main_unrolling/experiments/unrolling_WDN0122/FOS_pump_sched_flow/UnrollingModel/model.pickle'
        with open(model_path, 'rb') as handle:
            model = torch.load(handle)
            self.model = model.eval()

    def predict(self, X):
        # Return predictions for a batch of solutions
        return self.model.to(self.device).float()(X, 24).detach().cpu().numpy()