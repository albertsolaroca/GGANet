import numpy as np
import pandas as pd
import networkx as nx
from ipywidgets import widgets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score


class Dashboard:
    def __init__(self, real_heads_pd, predicted_heads_pd, G, node_locations):
        self.real_heads_pd = real_heads_pd
        self.predicted_heads_pd = predicted_heads_pd

        self.G = G
        self.f = None
        self.selected_node_x = 0
        self.selected_node_y = 0
        self.node_locations_with_value = node_locations

        # Add one to every element in list below
        self.node_indexes = list(nx.get_node_attributes(self.G, 'ID'))
        self.node_names = [str(i + 1) for i in self.node_indexes]

        # The following code is to correct the indexes of the nodes. Reservoirs do not
        # have values and should be treated as such. At the same time the arrays of heads
        # do not even contain the reservoirs. Thus, we need to correct the indexes of the
        # tanks or anything whose index is after the reservoirs
        # Get the index of False values from self.node_locations_with_value
        nan_index = np.where(self.node_locations_with_value == False)[0]

        self.index_corrector = np.zeros(len(self.node_locations_with_value))
        correct_factor = 0
        for i in range(len(self.index_corrector)):
            if i in nan_index:
                self.index_corrector[i] = None
                correct_factor += 1
            else:
                self.index_corrector[i] = correct_factor


    def create_plotly_figure(self):

        node_trace = self._get_network_scatter_trace()

        f = go.FigureWidget(make_subplots(rows=1, cols=3, specs=[[{}, {'colspan': 2}, None]],
                                          subplot_titles=(" ", "Comparison of hydraulic heads")))
        f.add_trace(node_trace, row=1, col=1)

        series = self.real_heads_pd[self.node_indexes[0]]

        x = list(series.index)
        y = series.values

        scatter_trace = self._get_real_scatter_trace(x, y)
        f.add_trace(scatter_trace, row=1, col=2)

        scatter_trace = self._get_predicted_scatter_trace()
        f.add_trace(scatter_trace, row=1, col=2)

        edge_trace = self._get_edge_trace()
        f.add_trace(edge_trace, row=1, col=1)

        selected_point = go.Scatter(x=[self.selected_node_x], y=[self.selected_node_y], marker_size=15,
                                    marker_symbol='x-open',
                                    marker_line_width=2, marker_color="midnightblue", showlegend=False, )
        f.add_trace(selected_point, row=1, col=1)

        trace = f.data[0]

        f.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        f.update_layout(width=1600, height=800, title="Distribution of real Head at hour 0")
        f.update_xaxes(showticklabels=False, col=1)
        f.update_yaxes(showticklabels=False, col=1)
        f.update_yaxes(title_text="Head [masl]", row=1, col=2)
        f.update_xaxes(title_text="Time [hours]", row=1, col=2)

        # create our callback function
        def update_point(trace, points, selector):
            try:
                correction_term = self.index_corrector[points.point_inds[0]]
                if ~np.isnan(correction_term):
                    real_time_series = self.real_heads_pd[self.node_indexes[points.point_inds[0]] - correction_term]
                    model_time_series = self.predicted_heads_pd[self.node_indexes[points.point_inds[0]] - correction_term]

                    f.data[1].x = list(real_time_series.index)
                    f.data[2].x = list(model_time_series.index)

                    f.data[1].y = real_time_series.values
                    f.data[2].y = model_time_series.values

                    f.data[4].x, f.data[4].y = [], []
                    f.data[4].x, f.data[4].y = points.xs, points.ys

                    node_name = self.node_indexes[points.point_inds[0]]

                    f.layout.annotations[1].update(text=f"Comparison of hydraulic head for node {node_name}")
            except Exception as e:
                pass

        trace.on_click(update_point)

        self.f = f

    def _get_predicted_scatter_trace(self):
        series = self.predicted_heads_pd[self.node_indexes[0]]

        x = list(series.index)
        y = series.values

        scatter_trace = go.Scatter(x=x,
                                   y=y,
                                   name="Our model",
                                   mode="lines+markers",
                                   line=dict(width=3),
                                   marker=dict(size=4, color="#6F1D77"),
                                   )

        return scatter_trace

    def _get_real_scatter_trace(self, x, y):
        scatter_trace = go.Scatter(x=x,
                                   y=y,
                                   name="real",
                                   mode="lines+markers",
                                   line=dict(width=3),
                                   marker=dict(size=4, color="#00A6D6"),
                                   )

        return scatter_trace

    def _get_network_scatter_trace(self):
        coordinates_df = pd.DataFrame.from_dict(nx.get_node_attributes(self.G, 'pos'), orient='index')

        x_coord = coordinates_df[0]
        y_coord = coordinates_df[1]
        self.selected_node_x = x_coord[0]
        self.selected_node_y = y_coord[0]

        node_signal = self.real_heads_pd.iloc[0, :]
        value = node_signal.values
        sizeref = 2. * max(value) / (2 ** 2)
        node_trace = go.Scatter(
            x=x_coord, y=y_coord,
            mode='markers',
            name='coordinates',
            hovertemplate='%{text}',
            text=self._get_text(value),
            marker_size=value - min(value),
            marker=dict(color=value, sizeref=sizeref, sizemin=1, colorscale='YlGnBu', cmax=1, cmin=0, showscale=True,
                        colorbar=dict(x=-0.05, ticklabelposition='outside'),
                        line=dict(width=1, color='DarkSlateGrey')),
            showlegend=False,
        )

        return node_trace

    def _get_edge_trace(self):
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = self.G.nodes[edge[0]]['pos']
            x1, y1 = self.G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False, )

        return edge_trace

    def _get_text(self, values):
        # These values do not include the reservoir because its value is not predicted. Thus, we need to add some NaNs
        # in the right positions
        # Get the index of False values from self.node_locations_with_value
        nan_index = np.where(self.node_locations_with_value == False)[0]
        # Create a new array with the same length as the original one but with NaNs in the right positions
        values = np.insert(values, nan_index, np.nan)
        return ['<b><br> Node ID: </b> {name} <br> <b>Value:</b> {value:.2f}'.format(name=self.node_names[i],
                                                                                     value=values[i]) for i in
                range(len(self.node_indexes))]

    def display_results(self):
        self.create_plotly_figure()
        hour_slider = widgets.IntSlider(
            value=0.0,
            min=0,
            max=len(self.real_heads_pd) - 1,
            step=1.0,
            description='Hour:',
            continuous_update=False,
            layout=widgets.Layout(width='50%')
        )

        textbox = widgets.Dropdown(
            description='Property:   ',
            value='Real Head',
            options=['Predicted Head', 'Real Head', 'Error', 'R2']
        )

        error_heads_pd = self.real_heads_pd - self.predicted_heads_pd

        r2 = r2_score(self.real_heads_pd.to_numpy(), self.predicted_heads_pd.to_numpy(), multioutput='raw_values')
        clipped_r2 = np.clip(r2, -10, 1)
        inverted_r2 = 1 / (clipped_r2 - min(clipped_r2) + 1e-6)

        # wet_r2 = utils.wet_r2_per_node(self.swmm_heads_pd, self.predicted_heads_pd, elevation)
        # inverted_wet_r2 = 1 / (wet_r2 - min(wet_r2) + 1e-6)
        #
        # dry_r2 = utils.dry_r2_per_node(self.swmm_heads_pd, self.predicted_heads_pd, elevation)
        # inverted_dry_r2 = 1 / (dry_r2 - min(dry_r2) + 1e-6)

        min_value_head = self.real_heads_pd.min().min()

        def response(change):

            t = hour_slider.value
            property = textbox.value
            self.f.update_layout(title=f'Distribution of {property} at hour {t}')

            if (property == 'Real Head'):

                real_signal_at_t = self.real_heads_pd.iloc[t, :]
                value = real_signal_at_t.values
                print("The min value is ", min(value), " and the value is ", value)
                sizeref = 2. * max(abs(value)) / (4 ** 2)
                self.f.update_traces(text=self._get_text(value), selector=dict(name="coordinates"))
                self.f.update_traces(marker=dict(color=value, size=value - min_value_head, sizeref=sizeref, sizemin=1,
                                                 colorscale='YlGnBu', cmax=2, cmin=0),
                                     selector=dict(name="coordinates"))
                self.f.update_yaxes(title_text="Head [masl]", row=1, col=2)

            if (property == 'Predicted Head'):

                predicted_signal_at_t = self.predicted_heads_pd.iloc[t, :]
                value = predicted_signal_at_t.values

                sizeref = 2. * max(abs(value)) / (4 ** 2)
                self.f.update_traces(text=self._get_text(value), selector=dict(name="coordinates"))
                self.f.update_traces(marker=dict(color=value, size=value - min_value_head, sizeref=sizeref, sizemin=1,
                                                 colorscale='YlGnBu', cmax=2, cmin=0),
                                     selector=dict(name="coordinates"))
                self.f.update_yaxes(title_text="Head [masl]", row=1, col=2)

            if (property == 'Error'):

                error_signal_at_t = abs(error_heads_pd.iloc[t, :])
                value = error_signal_at_t.values

                sizeref = 2. * max(abs(value)) / (5 ** 2)
                self.f.update_traces(marker=dict(color=value, size=value - min(value), sizeref=sizeref, sizemin=1,
                                                 colorscale='inferno_r'), selector=dict(name="coordinates"))
                self.f.update_traces(text=self._get_text(value), selector=dict(name="coordinates"))
                self.f.update_yaxes(title_text="Error [m]", row=1, col=2)

            if property == 'R2':

                sizeref = 2. * max(inverted_r2) / (6 ** 2)
                self.f.update_traces(
                    marker=dict(color=clipped_r2, size=inverted_r2, sizeref=sizeref, sizemin=3, colorscale='RdBu'),
                    selector=dict(name="coordinates"))
                self.f.update_traces(text=self._get_text(r2), selector=dict(name="coordinates"))


        textbox.observe(response, names="value")
        self.f.show()
        hour_slider.observe(response, names="value")
        # Create the vbox layout with a scrollbar
        vbox_layout = widgets.Layout(overflow_x='visible', width='1600px')  # Adjust height as needed

        return widgets.VBox([textbox, self.f, hour_slider], layout=vbox_layout)
