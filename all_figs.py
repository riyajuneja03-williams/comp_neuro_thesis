import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain
import fig_create
import pd_data

"""
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
T_vals = [10, 30]
for T in T_vals:
    fig_create.create_heatmap('train_rate', 'single_burst_rate', 'actual_rate', f"fr_heatmap_T{T}.png", T=T)
    fig_create.create_heatmap('train_rate', 'single_burst_rate', 'cv', f"cv_heatmap_T{T}.png", T)

frame_path = os.path.join('thesis', 'data_frame.csv')

# create histograms
for T in T_vals:
    fig_create.create_hist('actual_rate', f"rate_hist_T{T}.png", log_bool=False, T=T, frame_path=frame_path)
    fig_create.create_hist('cv', f"cv_hist_T{T}.png", log_bool=False, T=T, frame_path=frame_path)
    fig_create.create_hist('actual_rate', f"log_rate_hist_T{T}.png", log_bool=True, T=T, frame_path=frame_path)

# create indiv scatterplots
vars = ['predicted_burst_rate', 'D',
    'num_spikes', 'burst_firing_rate', 'avg_ISI_within_bursts', 'burst_rate', '%_spikes_in_burst', '%_time_spent_bursting', 'firing_rate_non_bursting', 'burst_firing_rate_inc',
    'ps_num_spikes', 'ps_burst_firing_rate', 'ps_avg_ISI_within_bursts', 'ps_burst_rate', 'ps_%_spikes_in_burst', 'ps_%_time_spent_bursting', 'ps_firing_rate_non_bursting', 'ps_burst_firing_rate_inc',
    'mi_num_spikes', 'mi_burst_firing_rate', 'mi_avg_ISI_within_bursts', 'mi_burst_rate', 'mi_%_spikes_in_burst', 'mi_%_time_spent_bursting', 'mi_firing_rate_non_bursting', 'mi_burst_firing_rate_inc',
    'logisi_num_spikes', 'logisi_burst_firing_rate', 'logisi_avg_ISI_within_bursts', 'logisi_burst_rate', 'logisi_%_spikes_in_burst', 'logisi_%_time_spent_bursting', 'logisi_firing_rate_non_bursting', 'logisi_burst_firing_rate_inc',
    'cma_num_spikes', 'cma_burst_firing_rate', 'cma_avg_ISI_within_bursts', 'cma_burst_rate', 'cma_%_spikes_in_burst', 'cma_%_time_spent_bursting', 'cma_firing_rate_non_bursting', 'cma_burst_firing_rate_inc',
]

for var in vars:
    for T in T_vals:
        fig_name = f"{var}_T{T}_scatterplot.png"
        fig_create.create_frcv_scatterplot(
            var=var,
            T=T,
            fig_name=fig_name,
            frame_path=frame_path
        )

fig_create.create_lin_reg_sp(
    T=T,
    fig_name = 'frcv_linreg_scatterplot',
    frame_path=frame_path
)

fig_create.compare_methods(40, 40)

fig_create.roc_curves()
"""

# PD figures
pd_frame_path = os.path.join('thesis', 'pd_data_frame.csv')
pd_df = pd.read_csv(pd_frame_path)

(pd_T, pd_N) = pd_data.return_params()

# raster plot
for i in range(0, pd_N):
    # get spike train
    pd_trains = []
    pd_train_dir = f"train_{i:03d}"
    pd_file_name = os.path.join('thesis', 'pd_data', pd_train_dir, 'spikes.txt')
    with open(pd_file_name, 'r') as file:
        for line in file:
            pd_trains.append(float(line))
    raster_path = os.path.join('thesis', 'pd_data', pd_train_dir)
    fig_create.raster_plot(pd_trains, raster_path)

# create histograms
pd_frame_path = os.path.join('thesis', 'pd_data_frame.csv')
fig_create.create_hist('actual_rate', f"pd_rate_hist.png", log_bool=False, T=pd_T, frame_path=pd_frame_path)
fig_create.create_hist('cv', f"pd_cv_hist.png", log_bool=False, T=pd_T, frame_path=pd_frame_path)
fig_create.create_hist('actual_rate', f"pd_log_rate_hist.png", log_bool=True, T=pd_T, frame_path=pd_frame_path)

# create indiv scatterplots
pd_vars = [
    'ps_num_spikes', 'ps_burst_firing_rate', 'ps_avg_ISI_within_bursts', 'ps_burst_rate', 'ps_%_spikes_in_burst', 'ps_%_time_spent_bursting', 'ps_firing_rate_non_bursting', 'ps_burst_firing_rate_inc',
    'mi_num_spikes', 'mi_burst_firing_rate', 'mi_avg_ISI_within_bursts', 'mi_burst_rate', 'mi_%_spikes_in_burst', 'mi_%_time_spent_bursting', 'mi_firing_rate_non_bursting', 'mi_burst_firing_rate_inc',
    'logisi_num_spikes', 'logisi_burst_firing_rate', 'logisi_avg_ISI_within_bursts', 'logisi_burst_rate', 'logisi_%_spikes_in_burst', 'logisi_%_time_spent_bursting', 'logisi_firing_rate_non_bursting', 'logisi_burst_firing_rate_inc',
    'cma_num_spikes', 'cma_burst_firing_rate', 'cma_avg_ISI_within_bursts', 'cma_burst_rate', 'cma_%_spikes_in_burst', 'cma_%_time_spent_bursting', 'cma_firing_rate_non_bursting', 'cma_burst_firing_rate_inc',
]

for var in pd_vars:
    fig_name = f"pd_{var}_scatterplot.png"
    fig_create.create_frcv_scatterplot(
    var=var,
        T=pd_T,
        fig_name=fig_name,
        frame_path=pd_frame_path
    )

fig_create.pd_compare_methods(100)

fig_create.create_lin_reg_sp(
    T=pd_T,
    fig_name = 'pd_frcv_linreg_scatterplot',
    frame_path=pd_frame_path
)