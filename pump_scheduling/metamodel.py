import torch

class MyMetamodel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_path = '../main_unrolling/experiments/unrolling_WDN0132/FOS_pump_sched_flow/UnrollingModel/model.pickle'
        with open(model_path, 'rb') as handle:
            model = torch.load(handle)
            self.model = model.eval()

    def predict(self, X):
        # Return predictions for a batch of solutions
        return self.model.to(self.device).float()(X, 24).detach().cpu().numpy()