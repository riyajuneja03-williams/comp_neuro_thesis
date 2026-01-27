import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain
import cma
import stats

(T, N, params, tau_ref, tau_burst) = synspiketrain.return_params()

for i, param in enumerate(params):
    for j in range(0, N):

        # get spike train
        trains = []
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'spikes.txt')
        with open(file_name, 'r') as file:
            for line in file:
                trains.append(float(line))
        
        # apply CMA
        cma_bursts = cma.cma_burst_detection(trains)
        
        # save detected bursts
        cma_path = os.path.join('thesis', param_name, train_name, 'cma_bursts.txt')
        with open(cma_path, "w") as file:
            for burst in cma_bursts:
                burst = np.array(burst)
                np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
                
        # calculate burst statistics for detected bursts
        _, cma_burststats = stats.calculate_statistics(trains, cma_bursts, param[0], T, param[1], param[2], param[3], tau_ref, tau_burst)
        burst_stats_file = 'cma_stats.txt'
        stats_path = os.path.join('thesis', param_name, train_name, 'cma_stats.txt')

        cma_burststats_dict = {
            "cma_num_spikes": cma_burststats['num_spikes'], 
            "cma_burst_firing_rate": cma_burststats['burst_firing_rate'], 
            "cma_avg_ISI_within_bursts": cma_burststats['avg_ISI_within_bursts'], 
            "cma_burst_rate": cma_burststats['burst_rate'], 
            "cma_%_spikes_in_burst": cma_burststats['%_spikes_in_burst'], 
            "cma_%_time_spent_bursting": cma_burststats['%_time_spent_bursting'], 
            "cma_firing_rate_non_bursting": cma_burststats['firing_rate_non_bursting'], 
            "cma_burst_firing_rate_inc": cma_burststats['burst_firing_rate_inc']
        }

        # write stats to file
        cma_burststats_array = np.array(list(cma_burststats_dict.items()), dtype=object)
        np.savetxt(stats_path, cma_burststats_array, fmt = "%s", delimiter = ":")
