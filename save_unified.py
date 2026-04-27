import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain

# load df
frame_path = os.path.join('thesis', 'data_frame.csv')
df = pd.read_csv(frame_path)

new_cols = ['unified_bursts', 'unified_num_spikes', 'unified_burst_firing_rate', 'unified_avg_ISI_within_bursts', 'unified_burst_rate', 'unified_%_spikes_in_burst', 'unified_%_time_spent_bursting', 'unified_firing_rate_non_bursting', 'unified_burst_firing_rate_inc']
df[new_cols] = np.nan

(D, T, N, params) = synspiketrain.return_params()

for i, param in enumerate(params):
    for j in range(0, N):

        row = i * N + j

        frame_data = {}
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'unified_stats.txt')

        # get burst stats
        if not os.path.exists(file_name):
            continue
        with open(file_name, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    frame_data[key.strip()] = value.strip()
        # save to dataframe
        for k, v in frame_data.items():
            df.loc[row, k] = float(v)

# save data frame to file
frame_path = os.path.join('thesis', 'data_frame.csv')
df.to_csv(frame_path, index=False)
