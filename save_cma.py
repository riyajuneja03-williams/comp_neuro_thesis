import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synthetic_saver

# load df
frame_path = os.path.join('thesis', 'data_frame.csv')
df = pd.read_csv(frame_path)

new_cols = ['cma_bursts', 'cma_num_spikes', 'cma_burst_firing_rate', 'cma_avg_ISI_within_bursts', 'cma_burst_rate', 'cma_%_spikes_in_burst', 'cma_%_time_spent_bursting', 'cma_firing_rate_non_bursting', 'cma_burst_firing_rate_inc']
df[new_cols] = np.nan

for i, param in enumerate(synthetic_saver.params):
    for j in range(0, synthetic_saver.N):

        row = i * synthetic_saver.N + j

        frame_data = {}
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'cma_stats.txt')

        # get burst stats
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
