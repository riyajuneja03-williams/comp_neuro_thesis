import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synthetic_saver
import maxinterval
import stats

param = synthetic_saver.params
T = synthetic_saver.T
tau_ref = synthetic_saver.tau_ref
tau_burst = synthetic_saver.tau_burst

for i, param in enumerate(synthetic_saver.params):
    for j in range(0, synthetic_saver.N):

        # get spike train
        trains = []
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'spikes.txt')
        with open(file_name, 'r') as file:
            for line in file:
                trains.append(float(line))
        
        # apply MI
        mi_bursts = maxinterval.max_interval(trains)
        
        # save detected bursts
        mi_path = os.path.join('thesis', param_name, train_name, 'mi_bursts.txt')
        with open(mi_path, "w") as file:
            for burst in mi_bursts:
                burst = np.asarray(burst)
                np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
                
        # calculate burst statistics for detected bursts
        _, mi_burststats = stats.calculate_statistics(trains, mi_bursts, param[0], T, param[1], param[2], param[3], tau_ref, tau_burst)
        burst_stats_file = 'mi_stats.txt'
        stats_path = os.path.join('thesis', param_name, train_name, 'mi_stats.txt')

        mi_burststats_dict = {
            "mi_num_spikes": mi_burststats['num_spikes'], 
            "mi_burst_firing_rate": mi_burststats['burst_firing_rate'], 
            "mi_avg_ISI_within_bursts": mi_burststats['avg_ISI_within_bursts'], 
            "mi_burst_rate": mi_burststats['burst_rate'], 
            "mi_%_spikes_in_burst": mi_burststats['%_spikes_in_burst'], 
            "mi_%_time_spent_bursting": mi_burststats['%_time_spent_bursting'], 
            "mi_firing_rate_non_bursting": mi_burststats['firing_rate_non_bursting'], 
            "mi_burst_firing_rate_inc": mi_burststats['burst_firing_rate_inc']
        }

        # write stats to file
        miburststats_array = np.array(list(mi_burststats_dict.items()), dtype=object)
        np.savetxt(stats_path, miburststats_array, fmt = "%s", delimiter = ":")
