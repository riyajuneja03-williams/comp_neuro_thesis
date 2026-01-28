import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain

(D, T, N, params) = synspiketrain.return_params()

# create dataframe
col_names = ['train_rate', 'predicted_burst_rate', 'single_burst_rate', 'D', 'T',
             'actual_rate', 'cv', 'ISI_dist', 
             'num_spikes', 'burst_firing_rate', 'avg_ISI_within_bursts', 'burst_rate', '%_spikes_in_burst', '%_time_spent_bursting', 'firing_rate_non_bursting', 'burst_firing_rate_inc']
df = pd.DataFrame(columns=col_names)

# save each spike train as a row in the dataframe
for i, param in enumerate(params):
    for j in range(0, N):
        frame_data = {}
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'metadata.txt')
        with open(file_name, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    frame_data[key.strip()] = value.strip()
        df.loc[len(df)] = frame_data

# save data frame to file
frame_path = os.path.join('thesis', 'data_frame.csv')
df.to_csv(frame_path, index=False)
