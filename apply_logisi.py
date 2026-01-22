import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synthetic_saver
import logisi
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
        
        # apply logISI
        logisi_bursts = logisi.log_isi(trains)
        
        # save detected bursts
        logisi_path = os.path.join('thesis', param_name, train_name, 'logisi_bursts.txt')
        with open(logisi_path, "w") as file:
            for burst in logisi_bursts:
                burst = np.array(burst)
                np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
                
        # calculate burst statistics for detected bursts
        _, logisi_burststats = stats.calculate_statistics(trains, logisi_bursts, param[0], T, param[1], param[2], param[3], tau_ref, tau_burst)
        burst_stats_file = 'logisi_stats.txt'
        stats_path = os.path.join('thesis', param_name, train_name, 'logisi_stats.txt')

        logisi_burststats_dict = {
            "logisi_num_spikes": logisi_burststats['num_spikes'], 
            "logisi_burst_firing_rate": logisi_burststats['burst_firing_rate'], 
            "logisi_avg_ISI_within_bursts": logisi_burststats['avg_ISI_within_bursts'], 
            "logisi_burst_rate": logisi_burststats['burst_rate'], 
            "logisi_%_spikes_in_burst": logisi_burststats['%_spikes_in_burst'], 
            "logisi_%_time_spent_bursting": logisi_burststats['%_time_spent_bursting'], 
            "logisi_firing_rate_non_bursting": logisi_burststats['firing_rate_non_bursting'], 
            "logisi_burst_firing_rate_inc": logisi_burststats['burst_firing_rate_inc']
        }

        # write stats to file
        logisi_burststats_array = np.array(list(logisi_burststats_dict.items()), dtype=object)
        np.savetxt(stats_path, logisi_burststats_array, fmt = "%s", delimiter = ":")
