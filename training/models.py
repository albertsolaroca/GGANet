import math

import numpy as np
import torch
from torch import nn
from torch.nn import Linear, Sequential


class Dummy():
    def __init__(self, pressures_idx):
        self.pressures_idx = pressures_idx
        pass

    def evaluate(self, Y):
        press_mean_per_row = torch.mean(Y[:, :self.pressures_idx], dim=1)
        # repeat mean to match dimensions of Y
        press_mean = press_mean_per_row.unsqueeze(1).repeat(1, self.pressures_idx)
        flow_mean_per_row = torch.mean(Y[:, self.pressures_idx:], dim=1)
        flow_mean_per_row[:] = torch.mean(flow_mean_per_row, dim=0)
        # repeat mean to match dimensions of Y
        flow_mean = flow_mean_per_row.unsqueeze(1).repeat(1, Y.shape[1] - self.pressures_idx)

        return torch.cat((press_mean, flow_mean), dim=1)


class MLPDynamicOnly(nn.Module):
    def __init__(self, num_outputs, hid_channels, indices, junctions, num_layers=6):
        super(MLPDynamicOnly, self).__init__()
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

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]
            input = torch.cat((pump_settings, demands), dim=1)
            output = self.main(input)
            predictions.append(output)

        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions

# MLP model including static data. The static data is concatenated to the input of the MLP
# It was found that performance does not improve compares to the MLPDynamicOnly model
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

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]
            input = torch.cat((static_features, pump_settings, demands), dim=1)
            output = self.main(input)
            predictions.append(output)

        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions


# Standard BaselineUnrolling model. The adaptations made along with the .out layer lead it to being closer to an
# MLP rather than a traditional unrolling model
class BaselineUnrolling(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6, hid_channels=None):
        super(BaselineUnrolling, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.num_tanks = indices['tanks'].stop - indices['tanks'].start
        self.num_heads = junctions + self.num_tanks
        self.num_reservoirs = indices['reservoirs'].stop - indices['reservoirs'].start
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.num_flows = indices['diameter'].stop - indices['diameter'].start
        self.num_base_heads = self.num_reservoirs + self.num_tanks
        self.num_blocks = num_layers
        self.n = 1.852

        self.static_feat_end = indices['diameter'].stop

        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)

        self.hid_HF = nn.ModuleList()
        self.hid_FH = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_FH.append(Sequential(
                Linear(self.num_base_heads + self.pump_number + self.demand_nodes + self.num_flows, self.num_heads),
                nn.ReLU()))
            self.hid_HF.append(Sequential(Linear(self.num_heads, self.num_flows), nn.ReLU()))

        self.out = Linear(self.num_heads, num_outputs)

    def forward(self, x, num_steps=1):

        d = x[:, self.indices['diameter']].float().view(-1, self.num_flows, 1)

        h0 = torch.cat((x[:, self.indices['reservoirs']], x[:, self.indices['tanks']]), dim=1)

        q = torch.mul(math.pi / 4, torch.pow(d, 2)).view(-1, self.num_flows).float()

        predictions = []
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            s = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]

            input = torch.cat((h0, s, q, pump_settings), dim=1)

            for j in range(self.num_blocks):
                h = self.hid_FH[j](input)
                q = q - self.hid_HF[j](h)

            # Append the prediction for the current time step
            prediction = self.out(h)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions

# An attempt to adapt the baseline unrolling model to predicting flows. Unsuccessful so far
class BaselineUnrollingFlows(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6, hid_channels=None):
        super(BaselineUnrollingFlows, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.num_tanks = indices['tanks'].stop - indices['tanks'].start
        self.num_reservoirs = indices['reservoirs'].stop - indices['reservoirs'].start
        self.num_heads = junctions + self.num_tanks
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.num_pipes = indices['diameter'].stop - indices['diameter'].start
        self.num_base_heads = self.num_reservoirs + self.num_tanks
        self.num_blocks = num_layers
        self.n = 1.852

        self.static_feat_end = indices['diameter'].stop

        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)
        self.num_flows = self.num_pipes + self.pump_number

        self.hid_HF = nn.ModuleList()
        self.hid_FH = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_FH.append(Sequential(
                Linear(self.num_base_heads + self.demand_nodes + self.num_flows, self.num_heads),
                nn.ReLU()))
            self.hid_HF.append(Sequential(Linear(self.num_heads, self.num_flows), nn.ReLU()))

        self.out = Linear(self.num_heads, num_outputs)

    def forward(self, x, num_steps=1):

        d = x[:, self.indices['diameter']].float().view(-1, self.num_pipes, 1)

        h0 = torch.cat((x[:, self.indices['reservoirs']], x[:, self.indices['tanks']]), dim=1)
        # h0  = x[:, self.indices['reservoirs']]

        q_init = torch.mul(math.pi / 4, torch.pow(d, 2)).view(-1, self.num_pipes).float()

        predictions = []
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            s = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]

            if step == 0:
                q = torch.cat((q_init, pump_settings), dim=1)
            else:
                q_copy = q.clone()
                q_copy[:, -self.pump_number:] = pump_settings
                q = q_copy

            input = torch.cat((h0, s, q), dim=1)

            for j in range(self.num_blocks):
                h = self.hid_FH[j](input)
                q = q - self.hid_HF[j](h)

            # Append the prediction for the current time step
            prediction = torch.cat((h, q), dim=1)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
        # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)

        return predictions


class UnrollingModelSimple(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6, hid_channels=None):
        super(UnrollingModelSimple, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.num_heads = junctions + indices['base_heads'].stop
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.num_flows = indices['diameter'].stop - indices['diameter'].start
        self.num_base_heads = indices['base_heads'].stop - indices['base_heads'].start
        self.num_blocks = num_layers
        self.static_feat_end = indices['diameter'].stop
        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)

        self.hidq0_h = Linear(self.num_flows, self.num_heads)  # 4.14
        self.hids_q = Linear(self.demand_nodes, self.num_flows)  # 4.6/4.10
        self.hidh0_h = Linear(self.num_base_heads, self.num_heads)  # 4.7/4.11
        self.hidh0_q = Linear(self.num_base_heads + self.pump_number, self.num_flows)  # 4.8/4.12
        self.hid_S = Sequential(Linear(indices['diameter'].stop - indices['diameter'].start, self.num_flows),
                                nn.ReLU())  # 4.9/4.13

        self.hid_hf = nn.ModuleList()
        self.hid_fh = nn.ModuleList()
        self.resq = nn.ModuleList()
        self.hidD_h = nn.ModuleList()
        self.hidD_q = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_hf.append(Sequential(Linear(self.num_heads, self.num_flows), nn.PReLU()))
            self.hid_fh.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.resq.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.hidD_q.append(Sequential(Linear(self.num_flows, self.num_flows)))
            self.hidD_h.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))

        self.out = Linear(self.num_flows, num_outputs)

    def forward(self, x, num_steps=1):
        # s is the demand and h0 is the heads (perhaps different when tanks are added)
        h0, d, edge_features = (x[:, self.indices['base_heads']].float(),
                                x[:, self.indices['diameter']].float(),
                                x[:, self.indices['diameter'].start:self.indices['diameter'].stop].float())

        res_h0_h, res_S_q = self.hidh0_h(h0), self.hid_S(
            edge_features)

        q = torch.mul(math.pi / 4, torch.pow(d, 2)).float()  # This is the educated "guess" of the flow
        res_q_h = self.hidq0_h(q)  # 4.14

        predictions = []
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            s = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]
            heads_and_pump = torch.cat((h0, pump_settings), dim=1)
            res_h0_q = self.hidh0_q(heads_and_pump)

            res_s_q = self.hids_q(s)

            for i in range(self.num_blocks):
                D_q = self.hidD_q[i](torch.mul(q, res_S_q))  # 4.16
                D_h = self.hidD_h[i](D_q)  # 4.17
                hid_x = torch.mul(D_q,
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

class UnrollingModel(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6, hid_channels=None):
        super(UnrollingModel, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.tanks = indices['tanks'].stop - indices['tanks'].start
        self.reservoirs = indices['reservoirs'].stop - indices['reservoirs'].start
        self.num_heads = junctions + self.tanks
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.num_flows = indices['diameter'].stop - indices['diameter'].start
        self.num_base_heads = self.reservoirs + self.tanks
        self.num_blocks = num_layers
        self.static_feat_end = indices['pump_schedules'].start
        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)

        self.hidq0_h = Linear(self.num_flows, self.num_heads)  # 4.14
        self.hids_q = Linear(self.demand_nodes, self.num_flows)  # 4.6/4.10
        self.hidh0_h = Linear(self.num_base_heads, self.num_heads)  # 4.7/4.11
        self.hidh0_q = Linear(self.num_base_heads, self.num_flows)  # 4.8/4.12
        self.hid_S = Sequential(Linear(indices['diameter'].stop - indices['diameter'].start, self.num_flows),
                                nn.ReLU())  # 4.9/4.13

        self.hid_hf = nn.ModuleList()
        self.hid_fh = nn.ModuleList()
        self.resq = nn.ModuleList()
        self.hidD_h = nn.ModuleList()
        self.hidD_q = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_hf.append(Sequential(Linear(self.num_heads, self.num_flows), nn.PReLU()))
            self.hid_fh.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.resq.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.hidD_q.append(Sequential(Linear(self.num_flows + self.pump_number, self.num_flows)))
            self.hidD_h.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))

        self.out = Linear(self.num_flows, num_outputs)

    def forward(self, x, num_steps=1):

        # s is the demand and h0 is the heads (perhaps different when tanks are added)
        reservoirs, tank_levels, d = (x[:, self.indices['reservoirs']].float(), x[:, self.indices['tanks']].float(),
                                      x[:, self.indices['diameter']].float())

        h0 = torch.cat((reservoirs, tank_levels), dim=1)

        res_h0_q, res_h0_h, res_S_q = self.hidh0_q(h0), self.hidh0_h(h0), self.hid_S(d)

        q = torch.mul(math.pi / 4, torch.pow(d, 2)).float()  # This is the educated "guess" of the flow
        res_q_h = self.hidq0_h(q)  # 4.14

        predictions = []
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            s = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]

            res_s_q = self.hids_q(s)

            for i in range(self.num_blocks):
                D_q = self.hidD_q[i](torch.cat((torch.mul(q, res_S_q), pump_settings), dim=1))
                D_h = self.hidD_h[i](D_q)
                hid_x = torch.mul(D_q, torch.sum(torch.stack([q, res_s_q, res_h0_q]), dim=0))
                h = self.hid_fh[i](hid_x)
                hid_x = self.hid_hf[i](torch.mul(torch.sum(torch.stack([h, res_h0_h, res_q_h]), dim=0), D_h))
                q = torch.sub(q, hid_x)
                res_q_h = self.resq[i](q)

            # Append the prediction for the current time step
            prediction = self.out(q)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
            # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)
        return predictions

class UnrollingModelN(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6, hid_channels=None):
        super(UnrollingModelN, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        self.num_tanks = indices['tanks'].stop - indices['tanks'].start
        self.num_heads = junctions + self.num_tanks
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.num_reservoirs = indices['reservoirs'].stop - indices['reservoirs'].start
        self.num_blocks = num_layers
        self.static_feat_end = indices['pump_schedules'].start
        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)
        self.num_flows = indices['diameter'].stop - indices['diameter'].start + self.pump_number

        self.hidq0_h = Linear(self.num_flows, self.num_heads)  # 4.14
        self.hids_q = Linear(self.demand_nodes, self.num_flows)  # 4.6/4.10
        self.hidh0_h = Linear(self.num_reservoirs + self.num_tanks, self.num_heads)  # 4.7/4.11
        self.hidh0_q = Linear(self.num_reservoirs + self.num_tanks, self.num_flows)  # 4.8/4.12
        self.hid_S = Sequential(Linear(indices['diameter'].stop - indices['diameter'].start, self.num_flows),
                                nn.ReLU())  # 4.9/4.13

        self.hid_hf = nn.ModuleList()
        self.hid_fh = nn.ModuleList()
        self.resq = nn.ModuleList()
        self.hidD_h = nn.ModuleList()
        self.hidD_q = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_hf.append(Sequential(Linear(self.num_heads, self.num_flows), nn.PReLU()))
            self.hid_fh.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.resq.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.hidD_q.append(Sequential(Linear(self.num_flows, self.num_flows)))
            self.hidD_h.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))

    def forward(self, x, num_steps=1):

        # s is the demand and h0 is the heads (perhaps different when tanks are added)
        reservoirs, tank_levels, d = (x[:, self.indices['reservoirs']].float(), x[:, self.indices['tanks']].float(),
                                      x[:, self.indices['diameter']].float())

        h0 = torch.cat((reservoirs, tank_levels), dim=1)

        res_h0_q, res_h0_h, res_S_q = self.hidh0_q(h0), self.hidh0_h(h0), self.hid_S(d)

        q_init = torch.mul(math.pi / 4, torch.pow(d, 2)).float()  # This is the educated "guess" of the flow

        predictions = []
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            s = x[:, timeseries_start:timeseries_end]


            pump_positions = [self.indices['pump_schedules'].start + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]

            pump_settings = x[:, pump_positions]

            res_s_q = self.hids_q(s)

            if step == 0:
                q = torch.cat((q_init, pump_settings), dim=1)
                res_q_h = self.hidq0_h(q)  # 4.14
            else:
                q_copy = q.clone()
                q_copy[:, -self.pump_number:] = pump_settings
                q = q_copy

            for i in range(self.num_blocks):
                D_q = self.hidD_q[i](torch.mul(q, res_S_q))
                D_h = self.hidD_h[i](D_q)
                hid_x = torch.mul(D_q, torch.sum(torch.stack([q, res_s_q, res_h0_q]), dim=0))
                h = self.hid_fh[i](hid_x)
                hid_x = self.hid_hf[i](torch.mul(torch.sum(torch.stack([h, res_h0_h, res_q_h]), dim=0), D_h))
                q = torch.sub(q, hid_x)
                res_q_h = self.resq[i](q)

            # Append the prediction for the current time step
            prediction = torch.cat((h, q), dim=1)
            # prediction = self.out(q)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
            # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)
        return predictions


class UnrollingModelQ(nn.Module):
    def __init__(self, num_outputs, indices, junctions, num_layers=6, hid_channels=None):
        super(UnrollingModelQ, self).__init__()
        torch.manual_seed(42)
        self.indices = indices
        # self.num_heads = junctions + indices['base_heads'].stop
        self.demand_nodes = junctions
        self.demand_start = indices['demand_timeseries'].start
        self.num_base_heads = indices['base_heads'].stop - indices['base_heads'].start
        self.num_blocks = num_layers
        self.static_feat_end = indices['pump_schedules'].start
        # To calculate amount of pumps we assume that the time period is 24
        self.pump_number = int((self.indices['pump_schedules'].stop - self.indices['pump_schedules'].start) / 24)
        self.num_flows = indices['diameter'].stop - indices['diameter'].start + self.pump_number
        self.num_heads = num_outputs - self.pump_number

        self.hidq0_h = Linear(self.num_flows, self.num_heads)  # 4.14
        self.hids_q = Linear(self.demand_nodes, self.num_flows)  # 4.6/4.10
        self.hidh0_h = Linear(self.num_base_heads, self.num_heads)  # 4.7/4.11
        self.hidh0_q = Linear(self.num_base_heads, self.num_flows)  # 4.8/4.12
        self.hid_S = Sequential(Linear(indices['diameter'].stop - indices['diameter'].start, self.num_flows),
                                nn.ReLU())  # 4.9/4.13

        self.hid_hf = nn.ModuleList()
        self.hid_fh = nn.ModuleList()
        self.resq = nn.ModuleList()
        self.hidD_h = nn.ModuleList()
        self.hidD_q = nn.ModuleList()

        for i in range(self.num_blocks):
            self.hid_hf.append(Sequential(Linear(self.num_heads, self.num_flows), nn.PReLU()))
            self.hid_fh.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.resq.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))
            self.hidD_q.append(Sequential(Linear(self.num_flows, self.num_flows)))
            self.hidD_h.append(Sequential(Linear(self.num_flows, self.num_heads), nn.ReLU()))

    def forward(self, x, num_steps=1):

        # s is the demand and h0 is the heads (perhaps different when tanks are added)
        h0, d = (x[:, self.indices['base_heads']].float(),
                 x[:, self.indices['diameter']].float())

        coeff_r, coeff_n = (x[:, self.indices['coeff_r']].float(),
                            x[:, self.indices['coeff_n']].float())

        res_h0_q, res_h0_h, res_S_q = self.hidh0_q(h0), self.hidh0_h(h0), self.hid_S(d)

        # This is the educated "guess" of the flow for the pipes
        q = torch.mul(math.pi / 4, torch.pow(d, 2)).float()
        # This is the educated "guess" of the flow for the pumps
        # pump_flows = torch.mul(coeff_n * coeff_r, torch.pow(q[:, 0:self.pump_number], coeff_n - 1))

        predictions = []
        for step in range(num_steps):
            demand_index_corrector = self.demand_nodes * step
            timeseries_start, timeseries_end = demand_index_corrector + self.demand_start, (
                    self.demand_start + demand_index_corrector + self.demand_nodes)
            s = x[:, timeseries_start:timeseries_end]

            pump_positions = [self.static_feat_end + (num_steps * pump) + step for pump in
                              list(range(self.pump_number))]
            pump_settings = x[:, pump_positions]

            if step == 0:
                q = torch.cat((q, pump_settings), dim=1)
                res_q_h = self.hidq0_h(q)  # 4.14

            res_s_q = self.hids_q(s)

            for i in range(self.num_blocks):
                D_q = self.hidD_q[i](torch.mul(q, res_S_q))
                if step > 0:
                    prev_pump_flows = q[:, -self.pump_number:]
                    prev_pump_flows = torch.clamp(prev_pump_flows, min=0)
                    new_pump_derivative = torch.mul(torch.mul(pump_settings, coeff_n * coeff_r),
                                                    torch.pow(prev_pump_flows, coeff_n - 1))
                    D_q_new = D_q.clone()
                    D_q_new[:, -self.pump_number:] = new_pump_derivative
                    D_q = D_q_new
                D_h = self.hidD_h[i](D_q)
                hid_x = torch.mul(D_q, torch.sum(torch.stack([q, res_s_q, res_h0_q]), dim=0))
                h = self.hid_fh[i](hid_x)
                hid_x = self.hid_hf[i](torch.mul(torch.sum(torch.stack([h, res_h0_h, res_q_h]), dim=0), D_h))
                q = torch.sub(q, hid_x)
                res_q_h = self.resq[i](q)

            # Append the prediction for the current time step
            pred_heads = h
            pred_pump_flows = q[:, -self.pump_number:]
            prediction = torch.cat((pred_heads, pred_pump_flows), dim=1)
            predictions.append(prediction)

        if num_steps == 1:
            return predictions[0]
            # Convert the list of predictions to a tensor
        predictions = torch.stack(predictions, dim=1)
        return predictions
