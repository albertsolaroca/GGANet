import math

import numpy as np
import torch
from torch import nn
from torch.nn import Linear, Sequential


class Dummy():
    def __init__(self):
        pass

    def evaluate(self, Y):
        mean_per_row = torch.mean(Y, dim=1)
        # repeat mean to match dimensions of Y
        mean = mean_per_row.unsqueeze(1).repeat(1, Y.shape[1])

        return mean


class MLPDemandsOnly(nn.Module):
    def __init__(self, num_outputs, hid_channels, indices, junctions, num_layers=6):
        super(MLPDemandsOnly, self).__init__()
        torch.manual_seed(42)
        self.hid_channels = hid_channels
        self.indices = indices
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.static_feat_end = indices['diameter'].stop
        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)

        self.total_input_length = self.pump_number + self.demand_nodes

        layers = [Linear(self.total_input_length, hid_channels),
                  nn.ReLU()]

        for l in range(num_layers - 1):
            layers += [Linear(hid_channels, hid_channels),
                       nn.ReLU()]

        layers += [Linear(hid_channels, num_outputs)]

        self.main = nn.Sequential(*layers)

    def forward(self, x, num_steps=1):

        predictions = []
        # static_features = x[:, :self.indices['demand_timeseries'].start]

        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            demands = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]
            input = torch.cat((pump_settings, demands), dim=1)
            output = self.main(input)
            predictions.append(output)

        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions


class MLPStatic(nn.Module):
    def __init__(self, num_outputs, hid_channels, indices, junctions, num_layers=6):
        super(MLPStatic, self).__init__()
        torch.manual_seed(42)
        self.hid_channels = hid_channels
        self.indices = indices
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.static_feat_end = indices['diameter'].stop

        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)

        self.total_input_length = self.static_feat_end + self.pump_number + self.demand_nodes

        layers = [Linear(self.total_input_length, hid_channels),
                  nn.ReLU()]

        for l in range(num_layers - 1):
            layers += [Linear(hid_channels, hid_channels),
                       nn.ReLU()]

        layers += [Linear(hid_channels, num_outputs)]

        self.main = nn.Sequential(*layers)

    def forward(self, x, num_steps=1):

        predictions = []
        static_features = x[:, :self.static_feat_end]
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                        self.demand_start + demand_index_corrector + self.demand_nodes)
            demands = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]
            input = torch.cat((static_features, pump_settings, demands), dim=1)
            output = self.main(input)
            predictions.append(output)

        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions


class MLPOld(nn.Module):
    def __init__(self, num_outputs, hid_channels, indices, num_layers=6):
        super(MLPOld, self).__init__()
        torch.manual_seed(42)
        self.hid_channels = hid_channels
        self.indices = indices

        self.total_input_length = indices[list(indices.keys())[-1]].stop

        layers = [Linear(self.total_input_length, hid_channels),
                  nn.ReLU()]

        for l in range(num_layers - 1):
            layers += [Linear(hid_channels, hid_channels),
                       nn.ReLU()]

        layers += [Linear(hid_channels, num_outputs)]

        # Add input of previous one and deamdn timeseries
        after_layers = [Linear(num_outputs, hid_channels),
                        nn.ReLU()]

        for l in range(num_layers - 1):
            after_layers += [Linear(hid_channels, hid_channels),
                             nn.ReLU()]

        after_layers += [Linear(hid_channels, num_outputs),
                         nn.ReLU()]

        self.start = nn.Sequential(*layers)
        self.main = nn.Sequential(*after_layers)

    def forward(self, x, num_steps=1):

        predictions = []

        x = self.start(x)
        predictions.append(x)
        for step in range(num_steps - 1):
            x = self.main(x)

            predictions.append(x)

        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions


# Define your LSTM model class
class LSTM(nn.Module):
    def __init__(self, num_outputs, hid_channels, indices, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hid_channels
        self.num_layers = num_layers
        self.output_size = num_outputs
        self.input_size = indices[list(indices.keys())[-1]].stop

        # Define the LSTM layer
        # Check why output size is included below
        self.lstm = nn.LSTM(self.input_size + self.output_size, self.hidden_size, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, static_input, num_steps=25):
        # Initialize hidden state with zeros
        # x = x.unsqueeze(1)
        batch_size = static_input.size(0)

        # Initial hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32).to(static_input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32).to(static_input.device)

        # Initialize the output sequence with zeros
        output_seq = torch.zeros(batch_size, num_steps, self.output_size, dtype=torch.float32).to(static_input.device)

        repeated_static_input = static_input.unsqueeze(1).repeat(1, 1, 1)
        # Iterate through time steps
        for t in range(num_steps):
            # Concatenate static_input with the previous output (if available)
            if t == 0:
                lstm_input = torch.cat((repeated_static_input, output_seq[:, t:t + 1, :]), dim=-1)
            else:
                lstm_input = torch.cat((repeated_static_input, output_seq[:, t - 1:t, :]), dim=-1)

            h0 = h0.to(torch.float32)
            c0 = c0.to(torch.float32)
            # Forward pass through the LSTM
            lstm_input = lstm_input.to(torch.float32)
            lstm_output, (h0, c0) = self.lstm(lstm_input, (h0, c0))

            # Predict the output for the current time step
            output_seq[:, t:t + 1, :] = self.linear(lstm_output)

        return output_seq.to(torch.float32)


class BaselineUnrolling(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6):
        super(BaselineUnrolling, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.num_heads = junctions
        self.num_flows = indices['diameter'].stop - indices['diameter'].start
        self.num_base_heads = indices['base_heads'].stop - indices['base_heads'].start
        self.num_blocks = num_layers
        self.n = 1.852

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.hid_HF = nn.ModuleList()
        self.hid_FH = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_HF.append(Sequential(Linear(self.num_heads, self.num_flows), nn.ReLU()))
            self.hid_FH.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))

        self.out = Linear(self.num_heads, num_outputs)

    def forward(self, x, num_steps=1):

        h0, d = torch.unsqueeze(x[:, self.indices['base_heads']], dim=2), \
            x[:, self.indices['diameter']].float().view(-1, self.num_flows, 1),

        q = torch.mul(math.pi / 4, torch.pow(d, 2)).view(-1, self.num_flows).float()

        predictions = []
        for step in range(num_steps):
            for j in range(self.num_blocks):
                h = self.hid_FH[j](q)
                q = q - self.hid_HF[j](h)

            # Append the prediction for the current time step
            prediction = self.out(h)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions


class UnrollingModel(nn.Module):
    def __init__(self, num_outputs, hid_channels, indices, junctions, num_layers=6):
        super(UnrollingModel, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.num_heads = indices['nodal_demands'].stop
        self.num_flows = indices['diameter'].stop - indices['diameter'].start
        self.num_base_heads = indices['base_heads'].stop - indices['base_heads'].start
        self.num_blocks = num_layers

        self.hidq0_h = Linear(self.num_flows, self.num_heads)  # 4.14
        self.hids_q = Linear(self.num_heads, self.num_flows)  # 4.6/4.10
        self.hidh0_h = Linear(self.num_base_heads, self.num_heads)  # 4.7/4.11
        self.hidh0_q = Linear(self.num_base_heads, self.num_flows)  # 4.8/4.12
        self.hid_S = Sequential(Linear(indices['diameter'].stop - indices['diameter'].start, self.num_flows),
                                nn.ReLU())  # 4.9/4.13

        # init.xavier_uniform_(self.hid_S[0].weight)
        # init.xavier_uniform_(self.hidq0_h.weight)
        # init.xavier_uniform_(self.hids_q.weight)
        # init.xavier_uniform_(self.hidh0_h.weight)
        # init.xavier_uniform_(self.hidh0_q.weight)

        self.hid_hf = nn.ModuleList()
        self.hid_fh = nn.ModuleList()
        self.resq = nn.ModuleList()
        self.hidD_h = nn.ModuleList()
        self.hidA_q = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_hf.append(Sequential(Linear(self.num_heads, self.num_flows), nn.PReLU()))
            self.hid_fh.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.resq.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.hidA_q.append(Sequential(Linear(self.num_flows, self.num_flows)))
            self.hidD_h.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
        # init.xavier_uniform_(self.hid_hf[i][0].weight)
        # init.xavier_uniform_(self.hid_fh[i][0].weight)
        # init.xavier_uniform_(self.resq[i][0].weight)
        # init.xavier_uniform_(self.hidA_q[i][0].weight)
        # init.xavier_uniform_(self.hidD_h[i][0].weight)

        self.out = Linear(self.num_flows, num_outputs)

    # init.xavier_uniform_(self.out.weight)

    def forward(self, x, num_steps=1):
        # s is the demand and h0 is the heads (perhaps different when tanks are added)
        s, h0, d, edge_features = (
            x[:, self.indices['nodal_demands']].float(), x[:, self.indices['base_heads']].float(),
            x[:, self.indices['diameter']].float(),
            x[:, self.indices['diameter'].start:self.indices['diameter'].stop].float())

        res_h0_q, res_s_q, res_h0_h, res_S_q = self.hidh0_q(h0), self.hids_q(s), self.hidh0_h(h0), self.hid_S(
            edge_features)

        q = torch.mul(math.pi / 4, torch.pow(d, 2)).float()  # This is the educated "guess" of the flow
        res_q_h = self.hidq0_h(q)  # 4.14

        predictions = []
        for step in range(num_steps):
            for i in range(self.num_blocks):
                A_q = self.hidA_q[i](torch.mul(q, res_S_q))  # 4.16
                D_h = self.hidD_h[i](A_q)  # 4.17
                hid_x = torch.mul(A_q,
                                  torch.sum(torch.stack([q, res_s_q, res_h0_q]), dim=0))  # 4.18 (inside parentheses)
                h = self.hid_fh[i](hid_x)  # 4.18
                hid_x = self.hid_hf[i](
                    torch.mul(torch.sum(torch.stack([h, res_h0_h, res_q_h]), dim=0), D_h))  # 4.19 (inside parentheses)
                q = torch.sub(q, hid_x)  # 4.19
                res_q_h = self.resq[i](q)  # 4.14

            # Append the prediction for the current time step
            prediction = self.out(q)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)
        return predictions
