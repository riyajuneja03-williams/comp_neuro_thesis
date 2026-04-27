import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import pd_data

# load df
frame_path = os.path.join('thesis', 'pd_data_frame.csv')
df = pd.read_csv(frame_path)

(T, N) = pd_data.return_params()

new_cols = [
    'actual_rate', 'cv', 'isi_dist', 
    'mi_bursts', 'mi_num_spikes', 'mi_burst_firing_rate', 'mi_avg_ISI_within_bursts', 'mi_burst_rate', 'mi_%_spikes_in_burst', 'mi_%_time_spent_bursting', 'mi_firing_rate_non_bursting', 'mi_burst_firing_rate_inc',
    'ps_bursts', 'ps_num_spikes', 'ps_burst_firing_rate', 'ps_avg_ISI_within_bursts', 'ps_burst_rate', 'ps_%_spikes_in_burst', 'ps_%_time_spent_bursting', 'ps_firing_rate_non_bursting', 'ps_burst_firing_rate_inc',
    'cma_bursts', 'cma_num_spikes', 'cma_burst_firing_rate', 'cma_avg_ISI_within_bursts', 'cma_burst_rate', 'cma_%_spikes_in_burst', 'cma_%_time_spent_bursting', 'cma_firing_rate_non_bursting', 'cma_burst_firing_rate_inc',
    'logisi_bursts', 'logisi_num_spikes', 'logisi_burst_firing_rate', 'logisi_avg_ISI_within_bursts', 'logisi_burst_rate', 'logisi_%_spikes_in_burst', 'logisi_%_time_spent_bursting', 'logisi_firing_rate_non_bursting', 'logisi_burst_firing_rate_inc',
    'unified_bursts', 'unified_num_spikes', 'unified_burst_firing_rate', 'unified_avg_ISI_within_bursts', 'unified_burst_rate', 'unified_%_spikes_in_burst', 'unified_%_time_spent_bursting', 'unified_firing_rate_non_bursting', 'unified_burst_firing_rate_inc'
]
df[new_cols] = np.nan
df['isi_dist'] = df['isi_dist'].astype(object)


for i in range (0, N):
    # get trains
    mi_frame_data = {}
    ps_frame_data = {}
    cma_frame_data = {}
    logisi_frame_data = {}
    unified_frame_data = {}
    stats_frame_data = {}
    train_dir = f"train_{i:03d}"
    mi_file_name = os.path.join('thesis', 'pd_data', train_dir, 'mi_stats.txt')
    ps_file_name = os.path.join('thesis', 'pd_data', train_dir, 'ps_stats.txt')
    cma_file_name = os.path.join('thesis', 'pd_data', train_dir, 'cma_stats.txt')
    logisi_file_name = os.path.join('thesis', 'pd_data', train_dir, 'logisi_stats.txt')
    unified_file_name = os.path.join('thesis', 'pd_data', train_dir, 'unified_stats.txt')
    stats_file_name = os.path.join('thesis', 'pd_data', train_dir, 'spike_stats.txt')

    # get burst stats
    file_name = os.path.join('thesis', 'pd_data', train_dir)
    if not os.path.isdir(file_name):
        continue

    with open(stats_file_name, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                stats_frame_data[key.strip()] = value.strip()
    
    with open(mi_file_name, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                mi_frame_data[key.strip()] = value.strip()
    
    with open(logisi_file_name, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                logisi_frame_data[key.strip()] = value.strip()
    
    with open(ps_file_name, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                ps_frame_data[key.strip()] = value.strip()

    with open(cma_file_name, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                cma_frame_data[key.strip()] = value.strip()
    
    with open(unified_file_name, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                unified_frame_data[key.strip()] = value.strip()

    # save to dataframe
    for k, v in mi_frame_data.items():
        df.loc[i, k] = float(v)
    
    for k, v in logisi_frame_data.items():
        df.loc[i, k] = float(v)
    
    for k, v in cma_frame_data.items():
        df.loc[i, k] = float(v)

    for k, v in ps_frame_data.items():
        df.loc[i, k] = float(v)

    for k, v in unified_frame_data.items():
        df.loc[i, k] = float(v)
    
    for k, v in stats_frame_data.items():
        if k == 'isi_dist':
            df.loc[i, k] = v
        else:
            df.loc[i, k] = float(v)

# save data frame to file
frame_path = os.path.join('thesis', 'pd_data_frame.csv')
df.to_csv(frame_path, index=False)