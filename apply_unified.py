import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain
import unified_method
import stats

(D, T, N, params) = synspiketrain.return_params()

for i, param in enumerate(params):
    for j in range(0, N):

        # get spike train
        trains = []
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'spikes.txt')
        path_name = os.path.join('thesis', param_name, train_name)
        if not os.path.isdir(path_name):
            continue
    
        with open(file_name, 'r') as file:
            for line in file:
                trains.append(float(line))
        
        metadata = {}
        meta_name = os.path.join('thesis', param_name, train_name, 'metadata.txt')
        with open(meta_name, 'r') as file:
            for line in file:
                if ':' not in line:
                    continue
                key, value = line.strip().split(':', 1)
                metadata[key] = value
        burst_rate = float(metadata['predicted_burst_rate'])

        # get T
        metadata = {}
        meta_name = os.path.join('thesis', param_name, train_name, 'metadata.txt')
        with open(meta_name, 'r') as file:
            for line in file:
                if ':' not in line:
                    continue
                key, value = line.strip().split(':', 1)
                metadata[key] = value

        # extract T (cast to float)
        T = float(metadata['T'])
        burst_rate = float(metadata['predicted_burst_rate'])
        
        # apply unified method
        unified_bursts = unified_method.unified_burst_detection(trains, T)
        
        # save detected bursts
        unified_path = os.path.join('thesis', param_name, train_name, 'unified_bursts.txt')
        with open(unified_path, "w") as file:
            for burst in unified_bursts:
                burst = np.array(burst)
                np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
                
        # calculate burst statistics for detected bursts
        _, unified_burststats = stats.calculate_statistics(trains, unified_bursts, param[0], param[1], param[2], burst_rate, param[3])
        burst_stats_file = 'unified_stats.txt'
        stats_path = os.path.join('thesis', param_name, train_name, 'unified_stats.txt')

        unified_burststats_dict = {
            "unified_num_spikes": unified_burststats['num_spikes'], 
            "unified_burst_firing_rate": unified_burststats['burst_firing_rate'], 
            "unified_avg_ISI_within_bursts": unified_burststats['avg_ISI_within_bursts'], 
            "unified_burst_rate": unified_burststats['burst_rate'], 
            "unified_%_spikes_in_burst": unified_burststats['%_spikes_in_burst'], 
            "unified_%_time_spent_bursting": unified_burststats['%_time_spent_bursting'], 
            "unified_firing_rate_non_bursting": unified_burststats['firing_rate_non_bursting'], 
            "unified_burst_firing_rate_inc": unified_burststats['burst_firing_rate_inc']
        }

        # write stats to file
        unified_burststats_array = np.array(list(unified_burststats_dict.items()), dtype=object)
        np.savetxt(stats_path, unified_burststats_array, fmt = "%s", delimiter = ":")