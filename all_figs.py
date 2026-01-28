import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain
import fig_create

frame_path = os.path.join('thesis', 'data_frame.csv')
df = pd.read_csv(frame_path)

(D, T, N, params) = synspiketrain.return_params()

# master raster plots
for i, param in enumerate(params):
    all_trains = []
    for j in range(0, N):
        trains = []
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'spikes.txt')
        with open(file_name, 'r') as file:
            for line in file:
                trains.append(float(line))
        all_trains.append(trains)
    master_path = os.path.join('thesis', param_name)
    fig_create.raster_plot(all_trains, master_path)

# create heatmaps
fig_create.create_heatmap('train_rate', 'single_burst_rate', 'actual_rate', 'fr_heatmap.png')
fig_create.create_heatmap('train_rate', 'single_burst_rate', 'cv', 'cv_heatmap.png')


# create histograms
fig_create.create_hist('actual_rate', 'rate_hist.png', log_bool=False)
fig_create.create_hist('cv', 'cv_hist.png', log_bool=False)
fig_create.create_hist('actual_rate', 'log_rate_hist.png', log_bool=True)

# create indiv scatterplots
T_vals = [10, 30]
per_T_vars = ["predicted_burst_rate", "D"]


other_vars = [
    'num_spikes', 'burst_firing_rate', 'avg_ISI_within_bursts', 'burst_rate', '%_spikes_in_burst', '%_time_spent_bursting', 'firing_rate_non_bursting', 'burst_firing_rate_inc',
    'ps_num_spikes', 'ps_burst_firing_rate', 'ps_avg_ISI_within_bursts', 'ps_burst_rate', 'ps_%_spikes_in_burst', 'ps_%_time_spent_bursting', 'ps_firing_rate_non_bursting', 'ps_burst_firing_rate_inc',
    'mi_num_spikes', 'mi_burst_firing_rate', 'mi_avg_ISI_within_bursts', 'mi_burst_rate', 'mi_%_spikes_in_burst', 'mi_%_time_spent_bursting', 'mi_firing_rate_non_bursting', 'mi_burst_firing_rate_inc',
    'logisi_num_spikes', 'logisi_burst_firing_rate', 'logisi_avg_ISI_within_bursts', 'logisi_burst_rate', 'logisi_%_spikes_in_burst', 'logisi_%_time_spent_bursting', 'logisi_firing_rate_non_bursting', 'logisi_burst_firing_rate_inc',
    'cma_num_spikes', 'cma_burst_firing_rate', 'cma_avg_ISI_within_bursts', 'cma_burst_rate', 'cma_%_spikes_in_burst', 'cma_%_time_spent_bursting', 'cma_firing_rate_non_bursting', 'cma_burst_firing_rate_inc',
]

for var in per_T_vars:
    for T in T_vals:
        fig_name = f"{var}_T{T}_scatterplot.png"
        fig_create.create_frcv_scatterplot(
            var=var,
            T=T,
            fig_name=fig_name
        )

for var in other_vars:
    fig_create.create_frcv_scatterplot(
        var=var,
        T=None,
        fig_name=f"{var}_scatterplot.png"
    )

fig_create.compare_methods(50, 50)