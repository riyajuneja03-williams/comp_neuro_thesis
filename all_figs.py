import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synthetic_saver
import fig_create

frame_path = os.path.join('thesis', 'data_frame.csv')
df = pd.read_csv(frame_path)

# raster plots
# for i, param in enumerate(synthetic_saver.params):
    # all_trains = []
    # for j in range(0, synthetic_saver.N):
        # trains = []
        # param_name = f'param_{i:04d}'
        # train_name = f'train_{j:03d}'
        # file_name = os.path.join('thesis', param_name, train_name, 'spikes.txt')
        # with open(file_name, 'r') as file:
            # for line in file:
                # trains.append(float(line))
        # all_trains.append(trains)
        # path = os.path.join('thesis', param_name, train_name)
        # fig_create.raster_plot(trains, path)
    # master_path = os.path.join('thesis', param_name)
    # fig_create.raster_plot(all_trains, master_path)

# create heatmaps
fig_create.create_heatmap('rate', 'burst_rate', 'actual_rate', 'fr_heatmap.png')
fig_create.create_heatmap('rate', 'burst_rate', 'cv', 'cv_heatmap.png')

# create histograms
fig_create.create_hist('actual_rate', 'rate_hist.png', log_bool=False)
fig_create.create_hist('cv', 'cv_hist.png', log_bool=False)
fig_create.create_hist('actual_rate', 'log_rate_hist.png', log_bool=True)

# create indiv scatterplots
vars = ['rate', 'prob_burst', 'prob_end', 
    'num_spikes', 'burst_firing_rate', 'avg_ISI_within_bursts', 'burst_rate', '%_spikes_in_burst', '%_time_spent_bursting', 'firing_rate_non_bursting', 'burst_firing_rate_inc',
    'ps_num_spikes', 'ps_burst_firing_rate', 'ps_avg_ISI_within_bursts', 'ps_burst_rate', 'ps_%_spikes_in_burst', 'ps_%_time_spent_bursting', 'ps_firing_rate_non_bursting', 'ps_burst_firing_rate_inc',
    'mi_num_spikes', 'mi_burst_firing_rate', 'mi_avg_ISI_within_bursts', 'mi_burst_rate', 'mi_%_spikes_in_burst', 'mi_%_time_spent_bursting', 'mi_firing_rate_non_bursting', 'mi_burst_firing_rate_inc'
]

for var in vars:
    fig_name = f"{var}_scatterplot.png"
    fig_create.create_frcv_scatterplot(var, fig_name)

# create 3x3 scatterplot
prob_vals = [0.2, 0.5, 0.8]
fig, axes = plt.subplots(3, 3, figsize=(9,9), sharex=True, sharey=True)
for i, pb in enumerate(prob_vals):
    for j, pe in enumerate(prob_vals):
        sub_df = df[(df.prob_burst == pb) & (df.prob_end == pe)]
        fig_create.create_frcv_scatterplot(var = None, fig_name = None, ax=axes[i, j], df=sub_df)
        if i == 0:
            axes[i, j].set_title(f"prob_end = {pe}")
        if j == 0:
            axes[i, j].set_ylabel(f"prob_burst = {pb}")
fig.supylabel("coefficient of variation")
fig.supxlabel("firing rate")
plt.tight_layout()
plt.savefig(os.path.join('thesis', 'fr_cv_probs_grid.png'))
plt.close()
        